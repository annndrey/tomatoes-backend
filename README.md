To build a docker image:

```
docker build  --rm -t tomatoesdemo .
docker run --name tomato_container -p 5455:5455 tomatoesdemo
```