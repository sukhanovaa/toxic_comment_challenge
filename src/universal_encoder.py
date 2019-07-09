import json
import tensorflow as tf
import tensorflow_hub as hub
from source.utils import take, import_file


def _parse(x):
    x = tf.cast(x,tf.string)
    return x


def _encoder(x_set, batch_size=100):
    dataset = tf.data.Dataset.from_tensor_slices(x_set)     
    dataset = dataset.map(lambda x : _parse(x)).batch(batch_size)
    dataset_iterator = dataset.make_one_shot_iterator()   
    return dataset_iterator.get_next()


def encode_batch(batch, config):
    #result dim -- x512
    input_lines = tf.placeholder(tf.string, shape=(None))
    with tf.Session(config=config) as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        embedded = session.run(USE(input_lines), feed_dict={input_lines: batch})
    return embedded


def encode_file(path, input_type='csv', field=None, tsv_delimiter=None,
                batch_size=1000, from_line=None, config=None):
    '''
    Encode sentences with USE by feeding the model batches.
    The preprocessing is not required by model
    '''
    reader = import_file(path, file_type=input_type, delimiter=tsv_delimiter)
    with open(path+'.emb', 'w') as out:
        if from_line:
            for i in range(from_line + 1):
                reader.readline()

        batch = take(batch_size, reader, field=field)
        count = 0
        while len(batch) > 0:
            embedded = encode_batch(batch, config=config)
            count +=1
            for e in embedded:
                #print(e)
                out.write(json.dumps(e.tolist())+'\n')
            print('{}'.format(count * batch_size))
            batch = take(batch_size, reader)


if __name__ == '__main__':
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  # Transformer version
    USE = hub.Module(module_url)
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.shape_optimization = 2  # hub compatibility issues