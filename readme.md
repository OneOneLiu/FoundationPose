# Reproduce readme

## Prepare code

1. Clone this repository
2. Build the docker image

```bash{.line-numbers}
cd docker/
docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
bash docker/run_container.sh
```
3. If it's the first time you launch the container, you need to build extensions.
```bash{.line-numbers}
bash build_all.sh
```

## Prepare data

