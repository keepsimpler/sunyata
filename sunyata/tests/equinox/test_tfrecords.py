import numpy
from sunyata.data.tfrecords import ReadTfrecordsFiles
from sunyata.utils import get_all_files_with_specific_filetypes_in_a_directory

import sys

def test_read_tfrecords_files():
    directory = "resources/openwebtext2/tfrecords/"

    tfrecords_files = get_all_files_with_specific_filetypes_in_a_directory(directory, filetypes=["tfrecords"])

    batch_size = 2
    chunk_size = 1024
    # chunks_per_file = 100000

    read_tfrecords_files = ReadTfrecordsFiles(tfrecords_files, batch_size, chunk_size)

    tfrecords_dataset = read_tfrecords_files.read()

    batch = tfrecords_dataset.next()

    assert batch['text'].shape == (batch_size, chunk_size)

    assert type(batch['text']) == numpy.ndarray
