import os


class FileIO:
    base_dir = os.getcwd()

    def __init__(self, base_dir=None):
        assert base_dir is None or isinstance(
            base_dir, str
        ), "base_dir must be a string or None"

        if base_dir is not None:
            self.base_dir = base_dir

        self.create_output_directory()

    def create_output_directory(self):
        """
        Create the output directory if it doesn't already exist.
        Note that this will overwrite files there if it already exists.
        """
        output_dir = os.path.join(self.base_dir, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
