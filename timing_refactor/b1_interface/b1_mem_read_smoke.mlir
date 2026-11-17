module attributes {ftlpu.icu_compression = "raw-macro"} {
  func.func @main() {
    ftlpu.command.mem {
      cycle = 0 : i64, queue = 0 : i64, opcode = "read",
      address = 64 : i64, packed_stream = 0 : i64,
      repeat_count = 4 : i64, repeat_interval = 1 : i64,
      address_stride = 1 : i64
    }
    return
  }
}
