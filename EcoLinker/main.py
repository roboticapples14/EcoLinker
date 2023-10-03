import argparse
import os
import restorationOptimizer
ecoscape = __import__("ecoscape-connectivity")

def main(args):
    # Reads and transltes the resistance dictionary.
    transmission_d = ecoscape.util.read_transmission_csv(args.permeability)

    optimizer = restorationOptimizer(
        habitat_fn=args.habitat,
        terrain_fn=args.terrain,
        permeability_dict=transmission_d,
        connectivity_fn=args.connectivity,
        flow_fn=args.flow,
        pixels=args.pixelsToRestore
    )

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--habitat', type=os.path.abspath, default=None,
                        help='Filename to a geotiff of the bird\'s habitat.')
    parser.add_argument('--terrain', type=os.path.abspath, default=None,
                        help='Filename to a geotiff of the terrain.')
    parser.add_argument('--permeability', type=os.path.abspath, default=None,
                        help='Filename to a CSV dictionary of the terrain permeability.'
                        'This should be a CSV with two columns, map_code, and transmission, the latter between 0 and 1.')
    parser.add_argument('--connectivity', type=os.path.abspath, default=None,
                        help='Filename to output geotiff file for connectivity.')
    parser.add_argument('--flow', type=os.path.abspath, default=None,
                        help='Filename to output geotiff file for flow. If missing, no flow is computed.')
    parser.add_argument('--pixelsToRestore', type=int, default=None,
                        help='Number of pixels able to be restored.')
    

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()