
import {
  pipeline,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let detector = null;

async function loadDetector() {
  if (detector) return detector;

  detector = await pipeline(
    "zero-shot-object-detection",
    "onnx-community/grounding-dino-tiny-ONNX",
    {
      device: "wasm",
      dtype: "uint8"
    }
  );

  return detector;
}

const button = document.getElementById("aiButton");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

button.addEventListener("click", async () => {
  const file = document.getElementById("photo").files?.[0];

  if (!file) {
    status.textContent = "Please upload a meal photo.";
    return;
  }

  button.disabled = true;
  results.replaceChildren();

  try {
    status.textContent = "Preparing your photo...";

    const image = await RawImage.fromBlob(file);
    const model = await loadDetector();

    const foods = [
      "cooked rice.",
      "grilled chicken.",
      "broccoli.",
      "carrots.",
      "potatoes.",
      "boiled eggs.",
      "salmon.",
      "fish.",
      "beef.",
      "pasta.",
      "bread.",
      "tomatoes.",
      "cucumber.",
      "lettuce.",
      "avocado.",
      "beans.",
      "cheese.",
      "pizza.",
      "sandwich.",
      "banana.",
      "apple.",
      "orange."
    ];

    const detections = [];

    for (let i = 0; i < foods.length; i++) {
      status.textContent =
        "Checking food " + (i + 1) +
        " of " + foods.length + "...";

      const found = await model(
        image,
        [foods[i]],
        { threshold: 0.25 }
      );

      detections.push(...found);
    }

    const unique = [];

    detections.forEach(item => {
      const label = item.label
        .toLowerCase()
        .replace(/\.$/, "");

      if (
        item.score >= 0.25 &&
        !unique.includes(label)
      ) {
        unique.push(label);
      }
    });

    const heading = document.createElement("h3");
    heading.textContent = "Detected food suggestions";
    results.appendChild(heading);

    if (unique.length === 0) {
      results.textContent =
        "No foods confidently detected. Please select your foods manually.";
    }

    unique.forEach(label => {
      const row = document.createElement("p");
      row.textContent = "✓ " + label;
      results.appendChild(row);
    });

    status.textContent =
      "Scan finished. Please confirm the detected foods.";

  } catch (error) {
    console.error(error);
    status.textContent = "Detection error: " + error.message;

  } finally {
    button.disabled = false;
  }
});
