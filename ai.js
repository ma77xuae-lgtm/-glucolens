

import {
  pipeline,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/grounding-dino-tiny-ONNX";
const MAX_IMAGE_SIZE = 768;
const CONFIDENCE_THRESHOLD = 0.25;

let detector = null;
let detectorPromise = null;

const button = document.getElementById("aiButton");
const photoInput = document.getElementById("photo");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

const foods = [
  "cooked rice",
  "grilled chicken",
  "broccoli",
  "carrots",
  "potatoes",
  "boiled eggs",
  "salmon",
  "fish",
  "beef",
  "pasta",
  "bread",
  "tomatoes",
  "cucumber",
  "lettuce",
  "avocado",
  "beans",
  "cheese",
  "pizza",
  "sandwich",
  "banana",
  "apple",
  "orange"
];

// Resize large photos to reduce browser memory usage.
async function preparePhoto(file) {
  let bitmap;

  try {
    if ("createImageBitmap" in window) {
      bitmap = await createImageBitmap(file);

      const scale = Math.min(
        1,
        MAX_IMAGE_SIZE / Math.max(bitmap.width, bitmap.height)
      );

      const width = Math.round(bitmap.width * scale);
      const height = Math.round(bitmap.height * scale);

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Could not prepare the photo.");
      }

      context.drawImage(bitmap, 0, 0, width, height);

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
    }

    // Fallback for browsers without createImageBitmap.
    const url = URL.createObjectURL(file);

    try {
      const image = new Image();

      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(
          new Error("Could not read this photo.")
        );
        image.src = url;
      });

      const scale = Math.min(
        1,
        MAX_IMAGE_SIZE / Math.max(image.naturalWidth, image.naturalHeight)
      );

      const canvas = document.createElement("canvas");
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);

      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Could not prepare the photo.");
      }

      context.drawImage(
        image,
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
      URL.revokeObjectURL(url);
    }

  } finally {
    if (bitmap) {
      bitmap.close();
    }
  }
}

// Load the AI model only once and reuse it.
async function loadDetector() {
  if (detector) {
    return detector;
  }

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

function showMessage(message) {
  status.textContent = message;
}

function showResults(detections) {
  results.replaceChildren();

  const uniqueFoods = [
    ...new Set(
      detections
        .filter(item => item.score >= CONFIDENCE_THRESHOLD)
        .map(item => item.label.toLowerCase().replace(/\.$/, ""))
    )
  ];

  const heading = document.createElement("h3");
  heading.textContent = "Detected food suggestions";
  results.appendChild(heading);

  if (uniqueFoods.length === 0) {
    const message = document.createElement("p");
    message.textContent =
      "No foods confidently detected. Please select your foods manually.";
    results.appendChild(message);
    return;
  }

  uniqueFoods.forEach(food => {
    const row = document.createElement("p");
    row.textContent = "✓ " + food;
    results.appendChild(row);
  });
}

if (!button || !photoInput || !status || !results) {
  console.error("GlucoLens: Required HTML elements were not found.");
} else {

  button.addEventListener("click", async () => {
    const file = photoInput.files?.[0];

    if (!file) {
      showMessage("Please upload a meal photo.");
      return;
    }

    if (!file.type.startsWith("image/")) {
      showMessage("Please choose a valid image.");
      return;
    }

    if (file.size > 30 * 1024 * 1024) {
      showMessage("This photo is too large. Please choose a smaller photo.");
      return;
    }

    button.disabled = true;
    results.replaceChildren();

    try {
      showMessage("Preparing your photo...");

      const image = await preparePhoto(file);

      showMessage(
        "Loading AI model. The first scan may take a few minutes..."
      );

      const model = await loadDetector();

      showMessage("Identifying foods. Please wait...");

      // Send all food labels together instead of running
      // a separate model call for every food.
      const detections = await model(
        image,
        foods,
        {
          threshold: CONFIDENCE_THRESHOLD,
          top_k: 30
        }
      );

      showResults(detections);

      showMessage(
        "Scan finished! Please confirm the detected foods."
      );

    } catch (error) {
      console.error("GlucoLens scanning error:", error);

      const message = error?.message || String(error);

      showMessage(
        "Could not scan this photo: " +
        message +
        " Please refresh the page and try again."
      );

    } finally {
      button.disabled = false;
    }
  });
}
