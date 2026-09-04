from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContainerContractTest(unittest.TestCase):
    def test_container_base_matches_repository_torch_floor(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            "pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime@sha256:",
            dockerfile,
        )
        self.assertNotIn("pytorch/pytorch:2.3.1", dockerfile)

    def test_container_installs_dependencies_in_an_isolated_venv(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            "python -m venv --system-site-packages /opt/sensevoice-venv",
            dockerfile,
        )
        self.assertIn("ENV PATH=/opt/sensevoice-venv/bin:$PATH", dockerfile)
        self.assertNotIn("--break-system-packages", dockerfile)

    def test_container_workflow_builds_prs_and_only_pushes_trusted_refs(self):
        workflow = (
            ROOT / ".github" / "workflows" / "sensevoice-container.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("pull_request:", workflow)
        self.assertIn("packages: write", workflow)
        self.assertIn("platforms: linux/amd64", workflow)
        self.assertIn("push: ${{ github.event_name != 'pull_request' }}", workflow)
        self.assertIn(
            "ghcr.io/${{ github.repository_owner }}/sensevoice",
            workflow,
        )

    def test_readme_uses_real_service_port_and_no_retired_registry(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("-p 50000:50000", readme)
        self.assertNotIn("60001", readme)
        self.assertNotIn(
            "registry.cn-hangzhou.aliyuncs.com/funasr/sensevoice",
            readme,
        )

    def test_verified_docker_run_is_identical_across_docs(self):
        gpu = (
            "docker run --rm --gpus all -p 50000:50000 "
            "-v sensevoice-models:/models sensevoice"
        )
        cpu = (
            "docker run --rm -e SENSEVOICE_DEVICE=cpu -p 50000:50000 "
            "-v sensevoice-models:/models sensevoice"
        )
        for name in ("README.md", "README_zh.md", "README_ja.md", "CONTRIBUTING.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn(gpu, text, msg=name)
            if name != "README_ja.md":
                self.assertIn(cpu, text, msg=name)

    def test_default_compose_starts_without_a_gpu_reservation(self):
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

        self.assertIn("50000:50000", compose)
        self.assertIn("sensevoice-models:/models", compose)
        self.assertNotIn("gpus:", compose)

    def test_api_uses_resolved_device_not_raw_auto(self):
        api = (ROOT / "api.py").read_text(encoding="utf-8")

        self.assertIn("resolve_sensevoice_device()", api)
        self.assertNotIn(
            'SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))',
            api,
        )

    def test_readmes_do_not_offer_private_ghcr_image_as_anonymous_pull(self):
        for name in ("README.md", "README_zh.md", "README_ja.md"):
            readme = (ROOT / name).read_text(encoding="utf-8").lower()

            self.assertNotIn(
                "docker pull ghcr.io/qwenaudio/sensevoice:latest",
                readme,
            )
            self.assertIn("docker build -t sensevoice .", readme)


if __name__ == "__main__":
    unittest.main()
