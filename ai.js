
import {
  pipeline,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID =
  "onnx-community/swin-finetuned-food101-ONNX";

const MAX_SIZE = 512;

const button = document.getElementById("aiButton");
const photoInput = document.getElementById("photo");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

let classifier = null;
let classifierPromise = null;

async function loadModel() {
  if (classifier) return classifier;

  if (!classifierPromise) {
    classifierPromise = pipeline(
      "image-classification",
      MODEL_ID,
      {
        device: "wasm",
        dtype: "q8"
      }
    ).then(model => {
      classifier = model;
      return model;
    }).catch(error => {
      classifierPromise = null;
      throw error;
    });
  }

  return classifierPromise;
}

async function preparePhoto(file) {
  const url = URL.createObjectURL(file);

  try {
    const img = new Image();

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(
        new Error("Could not open photo.")
      );
      img.src = url;
    });

    const scale = Math.min(
      1,
      MAX_SIZE / Math.max(
        img.naturalWidth,
        img.naturalHeight
      )
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
      throw new Error("Could not prepare photo.");
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
        result => result
          ? resolve(result)
          : reject(new Error("Photo compression failed.")),
        "image/jpeg",
        0.75
      );
    });

    return await RawImage.fromBlob(blob);

  } finally {
    URL.revokeObjectURL(url);
  }
}

function showResults(predictions) {
  results.replaceChildren();

  const heading = document.createElement("h3");
  heading.textContent = "AI food suggestions";
  results.appendChild(heading);

  predictions.forEach(item => {
    const row = document.createElement("p");

    const name = item.label
      .replace(/_/g, " ")
      .replace(/\b\w/g, letter => letter.toUpperCase());

    row.textContent =
      name + " — " + (item.score * 100).toFixed(1) + "%";

    results.appendChild(row);
  });

  const note = document.createElement("p");
  note.className = "small";
  note.textContent =
    "These are possible dish matches, not confirmed ingredients. Please check and select your foods manually.";

  results.appendChild(note);
}

button.addEventListener("click", async () => {
  const file = photoInput.files?.[0];

  if (!file) {
    status.textContent = "Please upload a food photo.";
    return;
  }

  if (!file.type.startsWith("image/")) {
    status.textContent = "Please select a valid image.";
    return;
  }

  button.disabled = true;
  results.replaceChildren();

  try {
    status.textContent = "Preparing photo...";

    const image = await preparePhoto(file);

    status.textContent =
      "Loading smaller food AI model...";

    const model = await loadModel();

    status.textContent = "Recognizing your food...";

    const predictions = await model(image, {
      top_k: 5
    });

    showResults(predictions);

    status.textContent =
      "Finished! Please confirm the food suggestions.";

  } catch (error) {
    console.error("GlucoLens error:", error);

    status.textContent =
      "Could not scan: " +
      (error.message || String(error));

  } finally {
    button.disabled = false;
  }
});
