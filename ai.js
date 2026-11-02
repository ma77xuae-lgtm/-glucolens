
import { pipeline, env } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

let foodModel = null;

async function loadFoodModel() {
  if (!foodModel) {
    foodModel = await pipeline(
      "image-classification",
      "onnx-community/swin-finetuned-food101-ONNX",
      { dtype: "q4" }
    );
  }
  return foodModel;
}

const button = document.getElementById("aiButton");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

if (button) {
  button.addEventListener("click", async () => {
    const photo = document.getElementById("photo");
    const preview = document.getElementById("preview");

    if (!photo.files || !photo.files[0]) {
      status.textContent = "Please upload a meal photo first.";
      return;
    }

    button.disabled = true;
    results.replaceChildren();

    try {
      status.textContent =
        "Loading the AI model. The first download may take a few minutes...";

      const model = await loadFoodModel();

      status.textContent = "AI is examining your photo...";

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
        "Suggestions ready! Check that the AI identified your food correctly.";

    } catch (error) {
      console.error(error);
      status.textContent =
        "The AI could not run. Check your internet connection or try another browser.";
    } finally {
      button.disabled = false;
    }
  });
}
