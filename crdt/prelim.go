package crdt

// Every internal piece for locally-created nested types already existed:
// ContentType, abstractType.detached(), the prelimFlusher contract, and the
// generic ContentType branch in item.integrate that links the child and
// replays its buffered ops. What it lacked was a way for code OUTSIDE the
// package to build a detached type (abstractType is unexported) and a way to
// insert one into a YArray as its own item (Insert batches plain values into
// a single ContentAny).
//
// Together with the ContentType branch in YMap.Get and the detached buffering
// in YMap.Set, this is what a nested document shape needs: for example a
// Jupyter notebook cell, which is a Y.Map holding a Y.Text.

// A note on locking, because it is easy to trip over: Doc.GetArray and
// Doc.GetMap take the document lock, and Transact already holds it. Resolve the
// container BEFORE opening the transaction:
//
//	root := doc.GetArray("cells")          // outside
//	doc.Transact(func(txn *Transaction) {
//		cell := NewMapPrelim()
//		src := NewTextPrelim()
//		src.Insert(txn, 0, "hello", nil)
//		cell.Set(txn, "source", src)
//		root.PushType(txn, cell)
//	})
//
// Accessors that read shared state (Keys, Has, ToJSON) are likewise not safe
// from inside a Transact callback.

// Buffered mutations replay call-by-call at attach: each buffered op emits its
// own item, where Yjs coalesces detached content (_prelimContent) and emits
// the net result. A multi-call build — two Pushes, a key set twice, a
// set-then-delete — therefore converges with Yjs but is not byte-identical to
// it. The conformance fixtures pin the shapes that are. Reads on a detached
// type (Len, Get, Keys) see none of the buffered state until attach.

// NewTextPrelim returns a DETACHED YText. Mutations are buffered until it is
// attached (via YMap.Set or YArray.PushType) and replayed then, so its items
// get clocks above the container item's — the ordering genuine Yjs produces.
func NewTextPrelim() *YText {
	t := &YText{}
	t.owner = t
	t.itemMap = make(map[string]*Item)
	return t
}

// NewMapPrelim returns a DETACHED YMap. Sets are buffered until attached.
func NewMapPrelim() *YMap {
	m := &YMap{}
	m.owner = m
	m.itemMap = make(map[string]*Item)
	return m
}

// NewArrayPrelim returns a DETACHED YArray.
func NewArrayPrelim() *YArray {
	a := &YArray{}
	a.owner = a
	a.itemMap = make(map[string]*Item)
	return a
}

// PushType appends a DETACHED shared type to the array as its own nested item.
// Plain values go through Push, which batches them into one ContentAny item; a
// nested type must occupy an item of its own, hence the separate entry point
// (the same reason YXmlFragment exposes InsertElement/InsertText).
//
// Placement mirrors Push: anchor after the last PHYSICAL item, tombstones
// included, matching Yjs's typeListPushGenerics.
func (a *YArray) PushType(txn *Transaction, st sharedType) {
	bt := st.baseType()
	if !bt.detached() {
		panic("crdt: PushType requires a detached type (use NewMapPrelim/NewTextPrelim)")
	}
	if a.detached() {
		a.pending = append(a.pending, func(txn *Transaction) { a.PushType(txn, st) })
		return
	}
	t := &a.abstractType

	var last *Item
	for it := t.start; it != nil; it = it.Right {
		last = it
	}

	// The walk ends at the physical tail, so there is never a right
	// neighbour: OriginRight stays nil and last is nil only when the list is
	// empty.
	var origin *ID
	if last != nil {
		end := last.ID.Clock + uint64(last.Content.Len()) - 1
		origin = &ID{Client: last.ID.Client, Clock: end}
	}

	item := &Item{
		ID:      ID{Client: txn.doc.clientID, Clock: txn.doc.store.NextClock(txn.doc.clientID)},
		Origin:  origin,
		Left:    last,
		Parent:  t,
		Content: NewContentType(bt),
	}
	// item.integrate sets bt.item, assigns bt.doc, and calls flushPrelim on
	// the owner — so buffered children materialise top-down from here.
	item.integrate(txn, 0)
}

// InsertType inserts a DETACHED shared type at logical position index
// (0 = prepend, Len() = append), as its own nested item.
//
// It is to Insert what PushType is to Push: Insert batches plain values into a
// single ContentAny item, which a nested type cannot share. Without this, the
// only way to place a nested type anywhere but the end is PushType followed by
// Move — and Move emits ContentMove, a ygo extension no other Yjs
// implementation decodes, so a peer such as pycrdt rejects the whole update.
//
// Placement mirrors Insert: leftNeighbourAt uses LIVE-index semantics (it skips
// tombstones), splitting the neighbour when the index falls inside it. That is
// the deliberate difference from PushType, which anchors after the last
// PHYSICAL item so a concurrent Yjs push converges the same way.
func (a *YArray) InsertType(txn *Transaction, index int, st sharedType) {
	bt := st.baseType()
	if !bt.detached() {
		panic("crdt: InsertType requires a detached type (use NewMapPrelim/NewTextPrelim)")
	}
	if a.detached() {
		a.pending = append(a.pending, func(txn *Transaction) { a.InsertType(txn, index, st) })
		return
	}
	t := &a.abstractType

	left, offset := t.leftNeighbourAt(index)
	if offset > 0 {
		splitItem(txn, left, offset)
		// left now holds the [0,offset) part; its Right is the new right half.
	}

	var origin *ID
	var originRight *ID
	if left != nil {
		end := left.ID.Clock + uint64(left.Content.Len()) - 1
		origin = &ID{Client: left.ID.Client, Clock: end}
		if left.Right != nil {
			id := left.Right.ID
			originRight = &id
		}
	} else if t.start != nil {
		id := t.start.ID
		originRight = &id
	}

	item := &Item{
		ID:          ID{Client: txn.doc.clientID, Clock: txn.doc.store.NextClock(txn.doc.clientID)},
		Origin:      origin,
		OriginRight: originRight,
		Left:        left,
		Parent:      t,
		Content:     NewContentType(bt),
	}
	// Signal the logical index for partial pos-cache invalidation, as Insert does.
	if index > 0 {
		t.insertHint = index
	}
	// item.integrate sets bt.item, assigns bt.doc, and calls flushPrelim on
	// the owner — so buffered children materialise top-down from here.
	item.integrate(txn, 0)
}
