// Deployment-tunable model location for the ProtonPottsMPNN page.
//
// The page reads three manifests from here: scoring/MANIFEST.json, ph/MANIFEST.json and decoder/MANIFEST.json, and
// the .onnx files they name. A same-origin ./models directory fits a checkout served locally or a deployment that
// bundled the weights in. A deployment that points at a remote release instead rewrites this ONE line after the
// build, as pages.yml does for the ProteinMPNN site. Nothing else in this page reads a model path.
export const MODEL_BASE = "./models";
