
import {
  pipeline,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let detector = null;

async function loadDetector() {
  if (detector) return detector;

  const model = "onnx-community/grounding-dino-tiny-ONNX";

  try {
    document.getElementById("aiStatus").textContent =
      "Loading AI model for your first scan...";

    detector = await pipeline(
      "zero-shot-object-detection",
      model,
      {
        device: "webgpu",
        dtype: "q4f16"
      }
    );
  } catch (gpuError) {
    console.warn("WebGPU unavailable. Trying CPU.", gpuError);

    detector = await pipeline(
      "zero-shot-object-detection",
      model,
      {
        device: "wasm",
        dtype: "uint8"
      }
    );
  }

  return detector;
}

const button = document.getElementById("aiButton");
const status = document.getElementById("aiStatus");
const results = document.getElementById("aiResults");

button.addEventListener("click", async () => {
  const input = document.getElementById("photo");
  const file = input.files?.[0];

  if (!file) {
    status.textContent = "Please upload a meal photo first.";
    return;
  }

  button.disabled = true;
  results.replaceChildren();

  try {
    status.textContent = "Preparing your photograph...";

    const image = await RawImage.fromBlob(file);
    const model = await loadDetector();

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

    status.textContent = "Looking for individual foods...";

    const detections = await model(
      image,
      foods,
      {
        threshold: 0.12,
        top_k: 15
      }
    );

    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    canvas.style.borderRadius = "12px";

    const ctx = canvas.getContext("2d");
    ctx.drawImage(
      await createImageBitmap(file),
      0, 0, image.width, image.height
    );

    const heading = document.createElement("h3");
    heading.textContent = "Food detection suggestions";
    results.appendChild(heading);

    const unique = [];

    detections.forEach(item => {
      const label = item.label.toLowerCase();

      if (!unique.includes(label) && item.score >= 0.12) {
        unique.push(label);

        const box = item.box;
        ctx.strokeStyle = "#1687df";
        ctx.lineWidth = Math.max(3, image.width / 180);
        ctx.strokeRect(
          box.xmin,
          box.ymin,
          box.xmax - box.xmin,
          box.ymax - box.ymin
        );

        ctx.fillStyle = "#1687df";
        ctx.font = `bold ${Math.max(14, image.width / 45)}px Arial`;
        ctx.fillText(label, box.xmin + 4, Math.max(18, box.ymin - 5));

        const row = document.createElement("p");
        row.textContent =
          label + " — model score " +
          (item.score * 100).toFixed(1) + "%";

        results.appendChild(row);
      }
    });

    if (unique.length === 0) {
      const message = document.createElement("p");
      message.textContent =
        "No confident food matches. Please select foods manually.";
      results.appendChild(message);
    } else {
      results.insertBefore(canvas, heading.nextSibling);
    }

    status.textContent =
      "Scan complete. Please confirm or correct every food.";

  } catch (error) {
    console.error(error);
    status.textContent =
      "Detection error: " + error.message;

  } finally {
    button.disabled = false;
  }
});
