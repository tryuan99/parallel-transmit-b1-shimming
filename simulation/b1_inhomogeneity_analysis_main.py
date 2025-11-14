from absl import app, flags, logging

from simulation.b_field import BField
from simulation.field_file import HFieldFile, SarFieldFile

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1, argv

    field = HFieldFile(FLAGS.data)
    mask = SarFieldFile(FLAGS.mask)
    b_field = BField(field, mask)
    logging.info("B1 inhomogeneity: %f", b_field.b1_inhomogeneity())
    b_field.plot_b1_magnitude()
    b_field.plot_normalized_b1_magnitude()


if __name__ == "__main__":
    flags.DEFINE_string("data", None, "H field file.")
    flags.DEFINE_string("mask", None, "SAR field file.")
    flags.mark_flags_as_required(["data", "mask"])

    app.run(main)
