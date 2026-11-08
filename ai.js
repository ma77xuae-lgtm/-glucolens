
import {
  pipeline,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/grounding-dino-tiny-ONNX";
const MAX_IMAGE_SIZE = 768;
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

// Load the model once and reuse it.
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

// Resize photos to reduce memory usage.
async function preparePhoto(file) {
  let bitmap;

  try {
    bitmap = await createImageBitmap(file);

    const scale = Math.min(
      1,
      MAX_IMAGE_SIZE / Math.max(bitmap.width, bitmap.height)
    );

    const canvas = document.createElement("canvas");

    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);

    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("Could not prepare the photo.");
    }

    context.drawImage(
      bitmap,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        result => result
          ? resolve(result)
          : reject(new Error("Photo conversion failed.")),
        "image/jpeg",
        0.85
      );
    });

    return await RawImage.fromBlob(blob);

  } finally {
    if (bitmap) bitmap.close();
  }
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
    results.appendChild(
      document.createTextNode(
        "No foods confidently detected. Please select your foods manually."
      )
    );
    return;
  }

  unique.forEach(label => {
    const row = document.createElement("p");
    row.textContent = "✓ " + label;
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
    status.textContent = "Please choose a valid image.";
    return;
  }

  if (file.size > 30 * 1024 * 1024) {
    status.textContent = "Photo is too large. Choose a smaller photo.";
    return;
  }

  button.disabled = true;
  results.replaceChildren();

  try {
    status.textContent = "Preparing your photo...";

    const image = await preparePhoto(file);

    status.textContent =
      "Loading AI model. First scan may take a few minutes...";

    const model = await loadDetector();

    const detections = [];

    // IMPORTANT:
    // This model expects one candidate label per call.
    // Do not pass all 22 labels together.

    for (let i = 0; i < foods.length; i++) {
      status.textContent =
        "Checking food " + (i + 1) +
        " of " + foods.length + "...";

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
      "Scan finished! Please confirm the detected foods.";

  } catch (error) {
    console.error("GlucoLens error:", error);

    status.textContent =
      "Could not scan photo: " +
      (error.message || String(error)) +
      ". Please refresh and try again.";

  } finally {
    button.disabled = false;
  }
});
