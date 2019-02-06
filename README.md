# Distributed TensorFlow

We want to create a cluster of servers and train TensorFlow models in a distributed fashion on that cluster.


## TODO:
* catch Ctrl + C Command to terminate ParamServer process

## Files:

* `./templates/00_mnist_replica.py` older version of distributed TF
* `./templates/00_between_graph_replication_async_mnist.py` online deploy guide of distributed TF (from: Feb 2019)



## Benchmark
compare training time for 1000 global steps

* 2 Param Servers, 2 Workers, op_parallelism_threads=os.cpu.count(): ````5 min 02 sec````
    Explanation: Parameter synchronization over WiFi slows down training process (15Mbit network traffic at peak)
* 1 Param Servers, 2 Workers, op_parallelism_threads=os.cpu.count(): ````2 min 42 sec````
* 1 Param Servers, 1 Workers (both on i7-2600k), op_parallelism_threads=8: ````3 min 02 sec````
* 1 Param Servers, 1 Workers (both on i5-MacBook Pro 2013), op_parallelism_threads=4: ````5 min 55 sec````

## Lessons Learned

* use tf.version >= 1.12 (with older tf versions I received errors during distributed execution)
* use LAN network, avoid Wifi

