
import {
  pipeline,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/grounding-dino-tiny-ONNX";
const MAX_SIZE = 512;
const THRESHOLD = 0.25;

let detector = null;
let detectorPromise = null;

const button = document.getElementById("aiButton");
const photoInput = document.getElementById("photo");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

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

// Compress and resize before AI processing.
async function preparePhoto(file) {
  const url = URL.createObjectURL(file);

  try {
    const img = new Image();

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(
        new Error("The photo could not be opened.")
      );
      img.src = url;
    });

    const scale = Math.min(
      1,
      MAX_SIZE / Math.max(img.naturalWidth, img.naturalHeight)
    );

    const canvas = document.createElement("canvas");

    canvas.width = Math.max(
      1,
      Math.round(img.naturalWidth * scale)
    );

    canvas.height = Math.max(
      1,
      Math.round(img.naturalHeight * scale)
    );

    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("Could not create photo canvas.");
    }

    context.drawImage(
      img,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        result => {
          if (result) resolve(result);
          else reject(new Error("Photo compression failed."));
        },
        "image/jpeg",
        0.65
      );
    });

    return await RawImage.fromBlob(blob);

  } finally {
    URL.revokeObjectURL(url);
  }
}

// Load the model once.
async function loadDetector() {
  if (detector) return detector;

  if (!detectorPromise) {
    detectorPromise = pipeline(
      "zero-shot-object-detection",
      MODEL_ID,
      {
        device: "wasm",
        dtype: "uint8"
      }
    ).then(model => {
      detector = model;
      return model;
    }).catch(error => {
      detectorPromise = null;
      throw error;
    });
  }

  return detectorPromise;
}

function displayResults(detections) {
  results.replaceChildren();

  const unique = [];

  detections.forEach(item => {
    const label = item.label
      .toLowerCase()
      .replace(/\.$/, "");

    if (
      item.score >= THRESHOLD &&
      !unique.includes(label)
    ) {
      unique.push(label);
    }
  });

  const heading = document.createElement("h3");
  heading.textContent = "Detected food suggestions";
  results.appendChild(heading);

  if (unique.length === 0) {
    const message = document.createElement("p");
    message.textContent =
      "No foods detected confidently. Please select foods manually.";
    results.appendChild(message);
    return;
  }

  unique.forEach(food => {
    const row = document.createElement("p");
    row.textContent = "✓ " + food;
    results.appendChild(row);
  });
}

button.addEventListener("click", async () => {
  const file = photoInput.files?.[0];

  if (!file) {
    status.textContent = "Please upload a meal photo.";
    return;
  }

  if (!file.type.startsWith("image/")) {
    status.textContent = "Please select a valid image.";
    return;
  }

  if (file.size > 30 * 1024 * 1024) {
    status.textContent = "Photo is too large. Choose a smaller photo.";
    return;
  }

  button.disabled = true;
  results.replaceChildren();

  try {
    status.textContent = "Preparing and compressing photo...";

    const image = await preparePhoto(file);

    status.textContent = "Loading AI model...";

    const model = await loadDetector();

    const detections = [];

    for (let i = 0; i < foods.length; i++) {
      status.textContent =
        "Checking food " + (i + 1) + " of " + foods.length + "...";

      const found = await model(
        image,
        [foods[i]],
        { threshold: THRESHOLD }
      );

      if (Array.isArray(found)) {
        detections.push(...found);
      }
    }

    displayResults(detections);

    status.textContent =
      "Scan complete! Please confirm the food suggestions.";

  } catch (error) {
    console.error("GlucoLens error:", error);

    status.textContent =
      "Scanning failed: " +
      (error.message || String(error));

  } finally {
    button.disabled = false;
  }
});
