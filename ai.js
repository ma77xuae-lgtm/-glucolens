

import { pipeline, env } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

let foodModel = null;

async function loadFoodModel() {
  if (!foodModel) {
    foodModel = await pipeline(
      "image-classification",
      "onnx-community/swin-finetuned-food101-ONNX",
      {
        dtype: "q4f16",
        device: "webgpu"
      }
    );
  }

  return foodModel;
}

const button = document.getElementById("aiButton");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

button.addEventListener("click", async () => {
  const photo = document.getElementById("photo");
  const preview = document.getElementById("preview");

  if (!photo.files || !photo.files[0]) {
    status.textContent = "Please upload a food photo first.";
    return;
  }

  button.disabled = true;
  results.replaceChildren();

  try {
    status.textContent =
      "Connecting to the AI model. Please wait...";

    const model = await loadFoodModel();

    status.textContent =
      "Analyzing your food photo...";

    const predictions = await model(preview, {
      top_k: 5
    });

    const heading = document.createElement("h3");
    heading.textContent = "Possible food matches";
    results.appendChild(heading);

    predictions.forEach(item => {
      const row = document.createElement("p");

      row.textContent =
        item.label.replaceAll("_", " ") +
        " · Model score: " +
        (item.score * 100).toFixed(1) +
        "%";

      results.appendChild(row);
    });

    status.textContent =
      "Done! Please check whether the food matches your photo.";
} catch (error) {
  console.error(error);

  status.textContent =
    "AI Error: " + (error.message || String(error));

} finally {
    button.disabled = false;

  }

});
