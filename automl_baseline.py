from ludwig.api import LudwigModel


def one_line_model(dataset_path: str, config_path: str, test_dataset_path: str):
    model = LudwigModel(config=config_path)
    results = model.train(dataset=dataset_path)
    eval_stats, _, _ = model.evaluate(dataset=test_dataset_path)
    return model, eval_stats, results
