import glob
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDIT_ROUTE_WORKFLOW = ROOT / "example_workflows" / "anima_t_reference_edit_route.json"


class WorkflowTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with EDIT_ROUTE_WORKFLOW.open("r", encoding="utf-8") as handle:
            cls.workflow = json.load(handle)
        cls.nodes = {node["id"]: node for node in cls.workflow["nodes"]}
        cls.nodes_by_type = {node["type"]: node for node in cls.workflow["nodes"]}
        cls.links = {link[0]: link for link in cls.workflow["links"]}

    def _node(self, node_type):
        try:
            return self.nodes_by_type[node_type]
        except KeyError:
            self.fail(f"Missing workflow node type: {node_type}")

    def _input_link(self, node, input_name):
        for item in node.get("inputs", []):
            if item["name"] == input_name:
                link_id = item.get("link")
                self.assertIsNotNone(link_id, f"{node['type']}.{input_name} is not linked")
                return self.links[link_id]
        self.fail(f"Missing workflow input: {node['type']}.{input_name}")

    def _source_node(self, node, input_name):
        link = self._input_link(node, input_name)
        return self.nodes[link[1]], link

    def test_edit_route_workflow_is_in_comfyui_template_folder(self):
        self.assertTrue(EDIT_ROUTE_WORKFLOW.exists())
        self.assertEqual(EDIT_ROUTE_WORKFLOW.parent.name, "example_workflows")

        comfyui_pattern = str(ROOT.parent / "*/example_workflows/*.json")
        matches = {Path(path).resolve() for path in glob.glob(comfyui_pattern)}
        self.assertIn(EDIT_ROUTE_WORKFLOW.resolve(), matches)

    def test_edit_route_workflow_contains_required_nodes(self):
        self.assertEqual(
            set(self.nodes_by_type),
            {
                "UNETLoader",
                "LoraLoaderModelOnly",
                "LoadImage",
                "VAELoader",
                "AnimaTReferenceEditRoute",
                "CLIPLoader",
                "CLIPTextEncode",
                "KSampler",
                "VAEDecode",
                "SaveImage",
            },
        )

    def test_edit_route_workflow_uses_animaedit_model_stack(self):
        unet = self._node("UNETLoader")
        lora = self._node("LoraLoaderModelOnly")
        clip = self._node("CLIPLoader")
        vae = self._node("VAELoader")

        self.assertEqual(unet["widgets_values"], ["animayume_v05.safetensors", "default"])
        self.assertEqual(lora["widgets_values"], ["Anima\\AnimaEditV1.safetensors", 1.0])
        self.assertEqual(clip["widgets_values"], ["qwen_3_06b_base.safetensors", "cosmos", "default"])
        self.assertEqual(vae["widgets_values"], ["qwen_image_vae.safetensors"])

    def test_edit_route_workflow_matches_t_reference_route(self):
        route = self._node("AnimaTReferenceEditRoute")
        sampler = self._node("KSampler")
        decode = self._node("VAEDecode")
        save = self._node("SaveImage")

        source, link = self._source_node(route, "model")
        self.assertEqual(source["type"], "LoraLoaderModelOnly")
        self.assertEqual(link[5], "MODEL")

        source, link = self._source_node(route, "image")
        self.assertEqual(source["type"], "LoadImage")
        self.assertEqual(link[5], "IMAGE")

        source, link = self._source_node(route, "vae")
        self.assertEqual(source["type"], "VAELoader")
        self.assertEqual(link[5], "VAE")

        source, link = self._source_node(sampler, "model")
        self.assertEqual(source["type"], "AnimaTReferenceEditRoute")
        self.assertEqual(link[2], 0)
        self.assertEqual(link[5], "MODEL")

        source, link = self._source_node(sampler, "latent_image")
        self.assertEqual(source["type"], "AnimaTReferenceEditRoute")
        self.assertEqual(link[2], 1)
        self.assertEqual(link[5], "LATENT")

        source, link = self._source_node(decode, "samples")
        self.assertEqual(source["type"], "KSampler")
        self.assertEqual(link[5], "LATENT")

        source, link = self._source_node(decode, "vae")
        self.assertEqual(source["type"], "VAELoader")
        self.assertEqual(link[5], "VAE")

        source, link = self._source_node(save, "images")
        self.assertEqual(source["type"], "VAEDecode")
        self.assertEqual(link[5], "IMAGE")

    def test_edit_route_workflow_uses_discussion_sampler_settings(self):
        sampler = self._node("KSampler")
        self.assertEqual(
            sampler["widgets_values"],
            [183041756463762, "fixed", 22, 3.4, "er_sde", "simple", 1.0],
        )


if __name__ == "__main__":
    unittest.main()
