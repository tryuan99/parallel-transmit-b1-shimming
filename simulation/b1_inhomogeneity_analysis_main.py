from absl import app, flags, logging

from simulation.b1_field_data import B1FieldData
from simulation.field_file import HFieldFile, SarFieldFile

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    field = HFieldFile(FLAGS.data)
    mask = SarFieldFile(FLAGS.mask)
    b1_field = B1FieldData(field, mask)
    logging.info(b1_field.data.head())
    logging.info("B1 inhomogeneity: %f", b1_field.calculate_inhomogeneity())
    b1_field.plot_magnitude()
    b1_field.plot_normalized_magnitude()


if __name__ == "__main__":
    flags.DEFINE_string("data", None, "H field file.")
    flags.DEFINE_string("mask", None, "SAR field file.")
    flags.mark_flags_as_required(["data", "mask"])

    app.run(main)
