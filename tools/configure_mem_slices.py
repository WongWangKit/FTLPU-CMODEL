#!/usr/bin/env python3
"""Resize the canonical MEM/SR chain in a shared LPU hardware configuration.

This changes physical topology only. The compiler resolves software placement
preferences with memory.automatic_layout; no model buffers are assigned here.
Custom routed fabrics must be configured explicitly rather than guessed.
"""
import argparse
import copy
import json
from pathlib import Path


def resize_mem(config, slices):
    result = copy.deepcopy(config)
    topology = result['stream_topology']
    bindings = topology['bindings']
    group = bindings['mem']['slices_per_group']
    if not isinstance(slices, int) or isinstance(slices, bool) or slices <= 0 or slices % group:
        raise ValueError(f'MEM slices must be a positive multiple of the {group}-slice SR group')
    old = config['mem']['slices_per_hemisphere']
    old_boundaries = [f'mem.b{i}' for i in range(old // group + 1)]
    new_boundaries = [f'mem.b{i}' for i in range(slices // group + 1)]
    fabric = topology['fabric_template']
    old_columns = old_boundaries + ['c2c_sxm', 'sxm_mxm']
    if (bindings['mem']['boundary_columns'] != old_boundaries
            or fabric['columns'] != old_columns
            or len(fabric['paths']) != 1
            or fabric['paths'][0]['columns'] != old_columns
            or fabric.get('routes')):
        raise ValueError('automatic resize requires the canonical linear MEM/SXM/MXM fabric')
    result['mem']['slices_per_hemisphere'] = slices
    bindings['mem']['boundary_columns'] = new_boundaries
    fabric['columns'] = new_boundaries + ['c2c_sxm', 'sxm_mxm']
    fabric['paths'][0]['columns'] = list(fabric['columns'])
    for binding in bindings['c2c'].values():
        if binding['column'] != old_boundaries[-1]:
            raise ValueError('automatic resize requires C2C at the last MEM boundary')
        binding['column'] = new_boundaries[-1]
    valid_columns = set(fabric['columns'])
    def validate_columns(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'column' and child not in valid_columns:
                    raise ValueError(f'resized topology would remove a bound column: {child}')
                validate_columns(child)
        elif isinstance(value, list):
            for child in value:
                validate_columns(child)
    validate_columns(topology)
    result.setdefault('memory', {})['automatic_layout'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--slices-per-hemisphere', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.base.resolve() == args.output.resolve():
        parser.error('--output must differ from --base to preserve the baseline configuration')
    result = resize_mem(json.loads(args.base.read_text(encoding='utf-8')),
                        args.slices_per_hemisphere)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'{args.output}: {args.slices_per_hemisphere} slices/hemisphere; '
          'software placement is selected by the C++ compiler')


if __name__ == '__main__':
    main()
