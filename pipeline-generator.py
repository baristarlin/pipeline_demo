from valohai import Pipeline

def main(config) -> Pipeline:

    pipe = Pipeline(name="utilspipeline", config=config)

    preprocess = pipe.execution("preprocess-dataset")
    train = pipe.execution("train-model")

    preprocess.output("preprocessed.mnist.npz").to(train.input("preprocessed_dataset"))

    return pipe
