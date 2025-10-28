#!/usr/bin/env python3
from scapy import __version__ as scapy_version
from scapy.all import rdpcap
# BLE layers in Scapy ≥ 2.5:
from scapy.layers.bluetooth4LE import BTLE, BTLE_ADV, BTLE_DATA, BTLE_CTRL
from scapy.layers.bluetooth import EIR_Hdr  # for parsing AdvData (EIR/AD structures)

PCAP = "out_ch37_t.pcap"

print(f"Using Scapy {scapy_version}")
pkts = rdpcap(PCAP)
print(f"Loaded {len(pkts)} packets from {PCAP}")

for i, pkt in enumerate(pkts, 1):
    print(f"\n=== Packet #{i} ===")
    # Full hierarchical dump:
    pkt.show()

    # If you want a compact BLE-specific summary too:
    if pkt.haslayer(BTLE):
        btle = pkt.getlayer(BTLE)
        print("  [BTLE] access_addr =", getattr(btle, "access_addr", None))

    if pkt.haslayer(BTLE_ADV):
        adv = pkt.getlayer(BTLE_ADV)
        advA = getattr(adv, "AdvA", None)
        advData = getattr(adv, "AdvData", b"") or b""
        print("  [ADV] PDU_type   =", getattr(adv, "PDU_type", None))
        print("  [ADV] AdvA (MAC) =", advA)
        print("  [ADV] AdvData    =", advData.hex())

        # Decode AD/EIR structures for readability
        i2 = 0
        while i2 < len(advData):
            ln = advData[i2]
            i2 += 1
            if ln == 0 or i2 + ln > len(advData):
                break
            t = advData[i2]
            v = advData[i2+1:i2+ln]
            i2 += ln
            if t == 0x01:
                print(f"    - Flags: 0x{int.from_bytes(v,'little'):02X}")
            elif t in (0x08, 0x09):
                try:
                    print(f"    - Name: {v.decode('utf-8','replace')}")
                except:
                    print(f"    - Name(raw): {v.hex()}")
            elif t == 0x16:
                if len(v) >= 2:
                    uuid = int.from_bytes(v[:2], "little")
                    print(f"    - Service Data 16-bit UUID 0x{uuid:04X}: {v[2:].hex()}")
                else:
                    print(f"    - Service Data (short): {v.hex()}")
            elif t == 0xFF:
                if len(v) >= 2:
                    cid = int.from_bytes(v[:2], "little")
                    print(f"    - Manufacturer (Company ID 0x{cid:04X}): {v[2:].hex()}")
                else:
                    print(f"    - Manufacturer: {v.hex()}")
            else:
                print(f"    - AD type 0x{t:02X}: {v.hex()}")

    elif pkt.haslayer(BTLE_DATA):
        data = pkt.getlayer(BTLE_DATA)
        print("  [DATA] LLID =", getattr(data, "LLID", None))

    elif pkt.haslayer(BTLE_CTRL):
        ctrl = pkt.getlayer(BTLE_CTRL)
        print("  [CTRL] opcode =", getattr(ctrl, "opcode", None))