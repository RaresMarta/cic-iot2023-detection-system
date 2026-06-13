"""CLI entrypoint.

  python -m ids.apps.monitor replay path/to.pcap [--realtime] [--loop]
  python -m ids.apps.monitor live --iface ids-br0

Sets environment config, then launches the FastAPI service with uvicorn.
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog='live_detector')
    ap.add_argument('--port', type=int, default=int(os.environ.get('IDS_PORT', '7870')))
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_replay = sub.add_parser('replay', help='replay a pcap file (offline/dev)')
    p_replay.add_argument('pcap')
    p_replay.add_argument('--realtime', action='store_true',
                          help='pace packets at their captured inter-arrival timing')
    p_replay.add_argument('--loop', action='store_true', help='replay forever')

    p_live = sub.add_parser('live', help='capture live from a Linux interface')
    p_live.add_argument('--iface', default=os.environ.get('IDS_IFACE', 'ids-br0'))

    sub.add_parser('simulate', help='replay real sampled CIC-IoT-2023 flows (offline demo)')

    args = ap.parse_args()

    if args.cmd == 'simulate':
        os.environ['IDS_SOURCE'] = 'simulate'
    elif args.cmd == 'replay':
        os.environ['IDS_SOURCE'] = 'replay'
        os.environ['IDS_PCAP'] = args.pcap
        if args.realtime:
            os.environ['IDS_PCAP_REALTIME'] = '1'
        if args.loop:
            os.environ['IDS_PCAP_LOOP'] = '1'
    elif args.cmd == 'live':
        os.environ['IDS_SOURCE'] = 'live'
        os.environ['IDS_IFACE'] = args.iface

    import uvicorn
    uvicorn.run('live_detector.service:app', host='0.0.0.0', port=args.port, log_level='info')


if __name__ == '__main__':
    main()
