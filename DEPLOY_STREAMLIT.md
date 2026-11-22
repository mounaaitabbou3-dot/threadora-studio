# Deploy THREADORA Studio on Streamlit Community Cloud (browser only)

Nothing has to be installed on your computer. Everything below happens in your browser.

## 1. Create the app
1. Go to https://share.streamlit.io and sign in with your GitHub account
   (allow access to private repositories when asked).
2. Click **Create app** → **Deploy a public app from GitHub** is not what you want; choose
   **Deploy from GitHub / "Yup, I have an app"**.
3. Fill in:
   - Repository: `mounaaitabbou3-dot/threadora-studio`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: pick a name you like (for example `threadora-yourname`).
4. Open **Advanced settings**:
   - Python version: **3.12** (the pinned packages were tested on 3.12).
   - Secrets: paste one line, with your own long password:
     ```
     THREADORA_PASSWORD = "choose-a-long-password-here"
     ```
5. Click **Deploy**. The first build installs Tesseract (from `packages.txt`) and the Python
   packages (from `requirements.txt`); it takes several minutes.

## 2. Keep it private
- Because the repository is private, the app is private: only you (and people you invite
  under **Share → Invite viewers** by email) can open it after signing in.
- The app also asks for `THREADORA_PASSWORD` before showing anything.
- Community Cloud allows **one private app per workspace**.

## 3. Use it
Open the app URL → enter the password → upload the image → choose the size → **Generate PDF**
→ **Download PDF**. Also download the **private audit files (ZIP)** for your records: the cloud
server does not keep orders after the app restarts or goes to sleep.

## Good to know
- The app sleeps after 12 hours without visitors; opening it wakes it in under a minute.
- Approximate resources: up to ~2.7 GB memory. A detailed 5×5 in design peaked at ~0.94 GB
  in testing. Only one pattern is generated at a time.
- To change the password: **App settings → Secrets**.
- Every push to `main` redeploys the app automatically.
