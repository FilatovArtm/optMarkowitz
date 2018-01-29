# optMarkowitz
Optimization methods for Markowitz portfolio problem

## Methods interface

```
def method(local_params, x_0, max_iter=100, trace=False)
```

## Experiments can be reproduced with Docker by the following command:

```
docker build -t opt_image .
docker run -v your_path:/optMarkowitz/results $IMAGEID
```
