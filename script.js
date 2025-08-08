let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById('recordBtn');
const player = document.getElementById('player');

recordBtn.onclick = async () => {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      player.src = audioUrl;
      player.style.display = 'block';
    };

    mediaRecorder.start();
    recordBtn.textContent = "⏹️ Stop";
  } else {
    mediaRecorder.stop();
    recordBtn.textContent = "🎙️ Record";
  }
};

