import numpy as np
import pandas as pd
from absl import app, flags, logging

from simulation.b_field import BField
from simulation.field_file import HFieldFile, SarFieldFile

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1, argv

    mask = SarFieldFile(FLAGS.mask)
    fields = [BField(HFieldFile(field), mask) for field in FLAGS.fields]

    df = pd.read_csv(FLAGS.shims, comment="#")
    coil_index_column, relative_magnitude_column, phase_column = df.columns
    weights = (df[relative_magnitude_column] *
               np.exp(1j * df[phase_column])).to_numpy()

    b_field = BField.sum(fields, weights)
    logging.info("B1 inhomogeneity: %f", b_field.b1_inhomogeneity())
    b_field.plot_b1_magnitude()
    b_field.plot_normalized_b1_magnitude()


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "fields",
        [
            "simulation/data/parallel_tx_phantom_b1_map_1.fld",
            "simulation/data/parallel_tx_phantom_b1_map_2.fld",
            "simulation/data/parallel_tx_phantom_b1_map_3.fld",
            "simulation/data/parallel_tx_phantom_b1_map_4.fld",
            "simulation/data/parallel_tx_phantom_b1_map_5.fld",
            "simulation/data/parallel_tx_phantom_b1_map_6.fld",
            "simulation/data/parallel_tx_phantom_b1_map_7.fld",
            "simulation/data/parallel_tx_phantom_b1_map_8.fld",
            "simulation/data/parallel_tx_phantom_b1_map_9.fld",
            "simulation/data/parallel_tx_phantom_b1_map_10.fld",
            "simulation/data/parallel_tx_phantom_b1_map_11.fld",
            "simulation/data/parallel_tx_phantom_b1_map_12.fld",
            "simulation/data/parallel_tx_phantom_b1_map_13.fld",
            "simulation/data/parallel_tx_phantom_b1_map_14.fld",
            "simulation/data/parallel_tx_phantom_b1_map_15.fld",
            "simulation/data/parallel_tx_phantom_b1_map_16.fld",
        ],
        "H field map files for each coil.",
    )
    flags.DEFINE_string("mask", "simulation/data/parallel_tx_phantom_sar.fld",
                        "SAR field file.")
    flags.DEFINE_string("shims",
                        "simulation/data/parallel_tx_phantom_shims.csv",
                        "B1 shim values.")

    app.run(main)
