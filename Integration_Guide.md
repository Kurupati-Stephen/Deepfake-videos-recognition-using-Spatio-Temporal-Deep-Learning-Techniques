# 🏭 Industrializing Your Forensic System for Live Calls

To make your project look like a professional, enterprise-grade forensic tool during live calls on Zoom, Google Meet, or WhatsApp, you should use **Virtual Broadcasting**. 

This allows you to either:
1. Broadcast your Streamlit Dashboard (with all the charts and live HUD) directly into your Zoom video feed.
2. Broadcast just your camera with the Forensic Bounding Boxes and Trust Scores overlaid.

The industry-standard way to achieve this is by using **OBS Studio**.

---

## 🛠️ Step 1: Install OBS Studio (Industry Standard)
OBS (Open Broadcaster Software) is used by professionals to route video feeds between applications.

1. Download and install [OBS Studio](https://obsproject.com/download) for macOS.
2. Open OBS Studio.

---

## 📺 Step 2: Capture Your Forensic Dashboard
We want to take your Streamlit Forensic Lab and make it your "Webcam".

1. In OBS, go to the **Sources** box at the bottom.
2. Click the **`+`** icon and select **Window Capture**.
3. Name it "Forensic Dashboard" and click OK.
4. In the Window dropdown, select your web browser that is running the Streamlit Dashboard (e.g., `[Google Chrome] Multimodal Forensic Lab`).
5. Click OK. You can now resize this window in the OBS preview to fit the entire screen perfectly.

> [!TIP]
> **Pro Tip:** Make your browser fullscreen (press `F11` or `Cmd+Ctrl+F` on Mac) so the Streamlit dashboard takes up the whole screen, hiding the browser tabs. This looks extremely professional and industrial.

---

## 🔴 Step 3: Start the Virtual Camera
OBS has a built-in feature that pretends to be a physical webcam.

1. On the right side of the OBS interface, in the **Controls** panel, click **Start Virtual Camera**.
2. OBS is now broadcasting your Streamlit dashboard as a webcam.

---

## 📞 Step 4: Route into Zoom / Google Meet / WhatsApp
Now, you just need to tell your calling app to use the OBS Virtual Camera instead of your laptop's camera.

### In Zoom:
1. Open Zoom and start or join a meeting.
2. Click the small up-arrow `^` next to the **Start Video / Stop Video** button.
3. Under "Select a Camera", choose **OBS Virtual Camera**.
4. You will now see your Forensic Dashboard being broadcasted to everyone in the meeting!

### In Google Meet:
1. Join a Google Meet call.
2. Click the three dots (⋮) at the bottom and go to **Settings** > **Video**.
3. Change the Camera to **OBS Virtual Camera**.

### In WhatsApp (Mac App):
1. Start a video call.
2. Click the Camera settings icon and select **OBS Virtual Camera**.

---

## 🔄 Two-Way Forensic Setup (The "God Mode")
To achieve the ultimate industrial setup during a live demonstration (like your project viva):

1. **You monitor them:** In your Streamlit app, set the `Forensic Input Source` to **Screen Content (Zoom/Meet/WhatsApp)**. This will analyze the other person's face on the call.
2. **They see the analysis:** Use the OBS Virtual Camera to broadcast your Streamlit Dashboard back to them.
3. **Result:** The professor/evaluator will see themselves on your screen, with your AI drawing bounding boxes over their face and calculating their live trust score in real-time.

> [!IMPORTANT]
> **Audio Routing:** Remember that if you want to analyze their audio too, you need to use **BlackHole** on your Mac. Set your Mac's speaker output to BlackHole so the audio routes into your Streamlit app, and check the "Select Audio Source" dropdown in the LIVE FORENSIC SUITE tab.
