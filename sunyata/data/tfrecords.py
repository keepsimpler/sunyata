from dataclasses import dataclass
from typing import List
import tensorflow as tf
from dataclasses import dataclass


@dataclass
class ReadTfrecordsFiles:
    tfrecords_files: List[str]
    batch_size: int
    chunk_size: int


    def read(self):
        tfrecords_dataset = tf.data.TFRecordDataset(self.tfrecords_files, num_parallel_reads=tf.data.AUTOTUNE)

        tfrecords_dataset = tfrecords_dataset.map(self.decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

        tfrecords_dataset = tfrecords_dataset.batch(self.batch_size, drop_remainder=True)  # batch must be *AFTER* map

        tfrecords_dataset = tfrecords_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        tfrecords_dataset = tfrecords_dataset.as_numpy_iterator()

        return tfrecords_dataset


    def decode_fn(self, record_bytes):
        return tf.io.parse_single_example(
            record_bytes,

            {
                "text": tf.io.FixedLenFeature((self.chunk_size,), dtype=tf.int64)
            }
        )