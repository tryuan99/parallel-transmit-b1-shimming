from absl import app, flags, logging

from simulation.field_file import HFieldFile

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    data = HFieldFile(FLAGS.data)
    logging.info(data.data.head())


if __name__ == "__main__":
    flags.DEFINE_string("data", None, "H field file.")
    flags.mark_flag_as_required("data")

    app.run(main)
