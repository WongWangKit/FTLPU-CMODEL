# FTLPU C2C first-stage patch

## Files to add

```text
include/ftlpu/c2c/
  c2c.hpp
  instruction.hpp
  link.hpp
  slice.hpp
  types.hpp

tests/c2c/
  c2c_test.cpp

examples/c2c/
  c2c_stream_demo.cpp
```

Append the two lines in `CMakeLists.increment.txt` to the root `CMakeLists.txt`.

## Cycle order

Use the following order in a multi-chip/system harness:

```cpp
sender_fabric.begin_cycle();
receiver_fabric.begin_cycle();

sender_tx.evaluate(sender_fabric, link);
receiver_rx.evaluate(receiver_fabric, link);

sender_topology.stage_active_routes(sender_fabric, sender_routes);
receiver_topology.stage_active_routes(receiver_fabric, receiver_routes);

sender_fabric.commit_cycle();
receiver_fabric.commit_cycle();

link.tick();
```

With `beat_bytes == 320`, a Send accepted in cycle N becomes available to a
Receive instruction in cycle N+1. The Receive stages twenty 16-byte segments
into the destination SR column; they become current after that cycle's global
fabric commit.

## Architectural assumptions in this first version

- One Send/Receive instruction transfers one 320-byte physical vector.
- TX requires all twenty tile segments to be valid and aligned in one cycle.
- TX performs exclusive consumption. The value therefore does not continue
  through a passive topology route in the same cycle.
- RX is a stream producer and never writes SRAM directly.
- A normal MEM Write instruction consumes the RX-produced West stream and
  writes the existing 320-byte SRAM row.
- `beat_bytes` is already parameterized. The initial configuration is 320,
  giving one-vector-per-cycle serialization.
