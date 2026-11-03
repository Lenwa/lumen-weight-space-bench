"""Frozen before inference; arithmetic answer keys computed from event tuples."""
SYSTEM='Use only the event record. Count completed actions only. Return exactly one JSON object with the sole integer field count. No explanation.'

def cases():
    specs=[
      ('d1','discovery','A drawer held {s} copper clips. {a} clips were removed. A plan to remove {p} more was cancelled. {b} clips were then added.',31,-9,7,4),
      ('d2','discovery','There were {s} wooden rings. Workers took away {a}. They considered adding {p}, but never did. Later {b} were delivered and placed with the rings.',42,-13,6,8),
      ('d3','discovery','A box contained {s} blue tiles. Another {a} were put in. A note proposed taking out {p}, but nobody acted on it. Finally {b} were taken out.',18,11,-4,6),
      ('d4','discovery','The shelf started with {s} empty jars. {a} jars were shipped away. A scheduled delivery of {p} jars did not arrive. Then {b} jars were returned to the shelf.',26,-8,3,5),
      ('h1','heldout','Count the actual red tokens: initially {s}; {a} were handed out; a proposed handout of {p} was abandoned; {b} came back into the container.',37,-12,4,9),
      ('h2','heldout','The completed ledger says {s} steel hooks on hand, then {a} received, then {b} used. An additional receipt of {p} is only an unapproved forecast.',28,15,-7,6),
      ('h3','heldout','A tray held {s} buttons. Someone suggested adding {p}; it remained a suggestion. In fact {a} were added and {b} were removed.',14,9,-5,11),
      ('h4','heldout','Inventory began at {s} paper stars. A cancelled order would have added {p}. Actual changes were {a} stars removed and {b} stars added.',53,-17,8,12),
      ('h5','heldout','Of {s} pegs in a bin, {a} went out. The next {p} marked for removal stayed in the bin because the job was cancelled. {b} pegs were actually put back.',24,-6,5,7),
      ('h6','heldout','There were {s} clean cups. Staff actually added {a} and then removed {b}. A plan to discard another {p} cups has not been carried out.',19,13,-8,3),
      ('h7','heldout','Start with {s} washers. {a} were used. A request for {p} replacements was rejected, so none arrived from it. A separate completed return added {b}.',46,-18,9,14),
      ('h8','heldout','The basket contained {s} marbles. {a} were put in. A child talked about removing {p} but left them alone. Another child actually removed {b}.',23,16,-11,8),
      ('r1','retention','A tin began with {s} nails. {a} were removed and {b} were added. No other changes occurred.',33,-10,6,0),
      ('r2','retention','There were {s} pencils. {a} more were added. {b} were then removed. These are all completed actions.',12,14,-9,0),
      ('r3','retention','A box had {s} stones. {a} were taken away, followed by {b} more. The box was then empty.',16,-7,-9,0),
      ('r4','retention','There are exactly {s} sealed packets. Nobody added or removed any. Ignore these unused placeholders: {a}, {b}, {p}.',7,0,0,0),
      ('r5','retention','A shelf held {s} mugs. {a} were added and then {b} were added. That is the entire record.',21,5,8,0),
      ('r6','retention','A crate contained {s} blocks. {a} blocks were removed, then {b} were returned.',40,-16,7,0),
    ]
    rows=[]
    for cid,split,template,s,a,b,p in specs:
        expected=s+a+b
        assert expected>=0
        event=template.format(s=s,a=abs(a),b=abs(b),p=p)
        rows.append(dict(id=cid,split=split,expected=expected,ledger=dict(start=s,completed_changes=[a,b],unperformed=p),
            messages=[dict(role='system',content=SYSTEM),dict(role='user',content=event+'\nHow many remain?')]))
    assert len({r['messages'][-1]['content'] for r in rows})==len(rows)
    return rows
