const video = document.getElementById("video");
const captureBtn = document.getElementById("captureBtn");
const snapshot = document.getElementById("snapshot");
const nameInput = document.getElementById("nameInput");
const resultCard = document.getElementById("resultCard");
const welcomeText = document.getElementById("welcomeText");
const facePreview = document.getElementById("facePreview");
const newBtn = document.getElementById("newBtn");

async function startCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    video.srcObject = stream;
  } catch (err) {
    alert("Could not access camera. Check permissions and refresh. " + err);
  }
}

function dataUrlFromVideo(){
  const w = video.videoWidth;
  const h = video.videoHeight;
  snapshot.width = w;
  snapshot.height = h;
  const ctx = snapshot.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);
  return snapshot.toDataURL("image/jpeg", 0.9); // compress a bit
}

captureBtn.addEventListener("click", async () => {
  captureBtn.disabled = true;
  captureBtn.textContent = "Processing...";

  const dataUrl = dataUrlFromVideo();
  const name = nameInput.value.trim();

  try {
    const resp = await fetch("/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl, name })
    });

    const result = await resp.json();

    if (!result.success) {
      alert(result.error || "Upload failed");
      captureBtn.disabled = false;
      captureBtn.textContent = "Capture & Welcome";
      return;
    }

    welcomeText.textContent = result.message;
    facePreview.src = result.face_url;
    resultCard.classList.remove("hidden");
    // small animation effect
    facePreview.animate([{ transform: "scale(0.7)", opacity: 0 }, { transform: "scale(1)", opacity: 1 }], { duration: 450, easing: "cubic-bezier(.2,.9,.2,1)" });

  } catch (err) {
    alert("Error contacting server: " + err);
  } finally {
    captureBtn.disabled = false;
    captureBtn.textContent = "Capture & Welcome";
  }
});

newBtn.addEventListener("click", () => {
  resultCard.classList.add("hidden");
  nameInput.value = "";
  nameInput.focus();
});

startCamera();
