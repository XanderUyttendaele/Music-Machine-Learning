import tensorflow as tf
import numpy as np
import random
song = np.load("E:\\musicdata\\Music-Machine-Learning\\song_data_training\\MAPS_MUS-alb_esp2_AkPnStgb.npy")
labels = np.load("E:\\musicdata\\Music-Machine-Learning\\song_data_labeled\\MAPS_MUS-alb_esp2_AkPnStgb.npy")
print("[INFO] Song loaded.")

length = song.shape[0]
songdata = []

sess = tf.Session()
saver = tf.train.import_meta_graph("E:\\musicdata\\Music-Machine-Learning\\saved_models\\model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint("E:\\musicdata\\Music-Machine-Learning\\saved_models\\"))

print("[INFO] Model restored.")
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
op = graph.get_tensor_by_name("prediction:0")
print("[INFO] Begin processing:")
for i in range(length):
    feed_dict = {x:[[song[i] for a in range(21)]]}
    songdata.append(sess.run(op, feed_dict)[0])
    print("[INFO] Processing timestep " + str(i+1))
print("[INFO] Picking random test")

index = random.randint(0, length-1)
print("[INFO] Index: " + str(index))
print("[INFO] Actual notes: " + str([i for i, e in enumerate(labels[index].tolist()) if e == 1.0]))
print("[INFO] Predictions: " + str([i for i, e in enumerate(songdata[index]) if e]))