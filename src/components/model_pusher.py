import sys
from huggingface_hub import HfApi
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Push the trained model to Hugging Face Hub.
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.hf_api = HfApi()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Ensuring Hugging Face repo exists...")
            # Create repo if it doesn't exist
            self.hf_api.create_repo(
                repo_id=self.model_pusher_config.hf_repo_id,
                repo_type="model",
                exist_ok=True,
                token=self.model_pusher_config.hf_token
            )

            logging.info("Uploading model to Hugging Face Hub...")
            # Upload the model
            self.hf_api.upload_file(
                path_or_fileobj=self.model_evaluation_artifact.trained_model_path,
                path_in_repo=self.model_pusher_config.hf_model_file_name,
                repo_id=self.model_pusher_config.hf_repo_id,
                repo_type="model",
                token=self.model_pusher_config.hf_token,
                commit_message="Add/update trained model"
            )

            # Direct download link
            model_url = f"https://huggingface.co/{self.model_pusher_config.hf_repo_id}/resolve/main/{self.model_pusher_config.hf_model_file_name}"

            model_pusher_artifact = ModelPusherArtifact(
                hf_repo_id=self.model_pusher_config.hf_repo_id,
                hf_model_file_name=self.model_pusher_config.hf_model_file_name,
                model_url=model_url
            )

            logging.info(f"Uploaded model successfully: {model_url}")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e
