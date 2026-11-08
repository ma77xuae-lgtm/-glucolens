
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

// Typical serving ranges in grams.
// These are general serving guides, NOT measurements
// calculated from the photograph.
const portionRanges = {
  rice: [180, 300],
  pasta: [200, 350],
  pizza: [150, 250],
  sandwich: [180, 300],
  salad: [120, 250],
  soup: [250, 450],
  chicken: [150, 280],
  seafood: [120, 250],
  meat: [150, 250],
  dessert: [80, 160],
  breakfast: [150, 300],
  snack: [80, 180],
  fruit: [100, 220],
  default: [100, 200]
};

function getPortionRange(label) {
  const name = label.toLowerCase();

  const groups = [
    ["rice", [
      "bibimbap", "fried_rice", "paella", "risotto"
    ]],
    ["pasta", [
      "gnocchi", "lasagna", "pad_thai", "ravioli",
      "spaghetti_bolognese", "spaghetti_carbonara"
    ]],
    ["pizza", ["pizza"]],
    ["sandwich", [
      "club_sandwich", "grilled_cheese_sandwich",
      "lobster_roll_sandwich", "pulled_pork_sandwich",
      "breakfast_burrito"
    ]],
    ["salad", [
      "caesar_salad", "greek_salad", "beet_salad",
      "caprese_salad", "seaweed_salad"
    ]],
    ["soup", [
      "clam_chowder", "french_onion_soup",
      "hot_and_sour_soup", "miso_soup",
      "pho", "ramen", "lobster_bisque"
    ]],
    ["chicken", [
      "chicken_curry", "chicken_wings",
      "chicken_quesadilla"
    ]],
    ["seafood", [
      "grilled_salmon", "fish_and_chips", "ceviche",
      "sashimi", "sushi", "scallops", "crab_cakes",
      "fried_calamari", "mussels", "oysters"
    ]],
    ["meat", [
      "steak", "filet_mignon", "prime_rib",
      "pork_chop", "beef_carpaccio", "beef_tartare"
    ]],
    ["dessert", [
      "cake", "pie", "tiramisu", "ice_cream",
      "cheesecake", "brownie", "donuts",
      "churros", "macarons", "baklava",
      "panna_cotta", "cannoli", "cup_cakes",
      "frozen_yogurt", "waffles"
    ]],
    ["breakfast", [
      "omelette", "eggs_benedict", "pancakes",
      "french_toast", "waffles"
    ]],
    ["snack", [
      "nachos", "french_fries", "onion_rings",
      "spring_rolls", "dumplings", "samosa",
      "edamame", "falafel"
    ]],
    ["fruit", [
      "apple"
    ]]
  ];

  for (const [category, labels] of groups) {
    if (labels.some(item => name.includes(item))) {
      return portionRanges[category];
    }
  }

  return portionRanges.default;
}

async function loadModel() {
  if (classifier) return classifier;

  if (!classifierPromise) {
    classifierPromise = pipeline(
      "image-classification",
      MODEL_ID,
      { device: "wasm", dtype: "q8" }
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
      img.onerror = () =>
        reject(new Error("Could not open photo."));
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

function createElement(tag, text, style = {}) {
  const element = document.createElement(tag);
  element.textContent = text;
  Object.assign(element.style, style);
  return element;
}

function showResults(predictions) {
  results.replaceChildren();

  results.appendChild(
    createElement("h3", "AI food analysis")
  );

  results.appendChild(
    createElement(
      "p",
      "Possible dish matches. These are alternatives, not separate foods.",
      { fontSize: "13px", color: "#64829b" }
    )
  );

  predictions.forEach((item, index) => {
    const range = getPortionRange(item.label);

    const card = document.createElement("div");

    Object.assign(card.style, {
      background: "#f2f8fd",
      borderRadius: "12px",
      padding: "14px",
      margin: "10px 0"
    });

    const name = item.label
      .replace(/_/g, " ")
      .replace(/\b\w/g, letter => letter.toUpperCase());

    card.appendChild(
      createElement(
        "strong",
        (index === 0 ? "Top match: " : "Alternative: ") + name,
        { display: "block", marginBottom: "8px" }
      )
    );

    card.appendChild(
      createElement(
        "div",
        range[0] + "–" + range[1] + " g",
        {
          fontSize: "25px",
          fontWeight: "bold",
          color: "#367db5"
        }
      )
    );

    card.appendChild(
      createElement(
        "p",
        "Typical serving estimate",
        {
          fontSize: "12px",
          color: "#64829b",
          margin: "5px 0 0"
        }
      )
    );

    card.appendChild(
      createElement(
        "p",
        "Model score: " +
          (item.score * 100).toFixed(1) + "%",
        {
          fontSize: "12px",
          color: "#64829b",
          margin: "5px 0 0"
        }
      )
    );

    results.appendChild(card);
  });

  results.appendChild(
    createElement(
      "p",
      "Important: Gram ranges are typical serving guides, not measurements from your photo. Confirm the food and adjust its amount manually before calculating nutrition.",
      {
        fontSize: "12px",
        color: "#64829b",
        lineHeight: "1.5"
      }
    )
  );
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

    status.textContent = "Loading food AI model...";

    const model = await loadModel();

    status.textContent = "Recognizing your food...";

    const predictions = await model(image, {
      top_k: 3
    });

    if (!predictions || predictions.length === 0) {
      throw new Error("No food predictions returned.");
    }

    showResults(predictions);

    status.textContent =
      "Analysis complete! Please confirm your food.";

  } catch (error) {
    console.error("GlucoLens error:", error);

    status.textContent =
      "Could not analyze photo: " +
      (error.message || String(error));

  } finally {
    button.disabled = false;
  }
});
