# Running Rupert's training on RunPod: step by step

**Money safety, read first**
- RunPod is pay-up-front. You add credit; when it runs out, machines stop. It cannot charge your card more **unless you turn on auto-pay. Leave auto-pay OFF.**
- An RTX 4090 costs about **$0.34/hour (~27p)** on Community Cloud.
- Our script deletes the machine itself when it finishes, crashes or hits its time limit.
- Worst realistic test run: about 3½ hours of machine time = **about £1**.
- **Always check the Pods page is empty when you're done.** That is the one habit that protects your credit.

## One-off setup (about 5 minutes)
1. Go to **runpod.io** and sign up.
2. Click **Billing** on the left. Add **$10** of credit. You enter the card details yourself.
3. On the same page, make sure **auto-pay / automatic top-up is OFF**.

## Each run
1. Click **Pods** on the left, then **Deploy** (or **+ Deploy**).
2. Pick **Community Cloud** (cheaper).
3. Find the **filter for CUDA version** and choose **13.0** (or the highest offered). *If you skip this, the script will refuse to run on an old machine and stop within minutes. That costs pennies but wastes your time.*
4. Choose **RTX 4090**. If none are available, an **RTX 5090** or **L40S** also works but costs more. Check the hourly price shown.
5. Template: anything called **RunPod PyTorch** (it includes "Jupyter"). Leave the disk sizes as they are.
6. Make sure it says **On-Demand** (not Spot/Interruptible), then click **Deploy**.
7. Wait until the pod shows **Running** (a minute or two). Click **Connect**, then **Jupyter Lab** (or "Connect to HTTP Service [Port 8888]").
8. In Jupyter: **File > New > Terminal**.
9. Copy this whole line, paste it into the terminal (right-click > Paste, or Ctrl+Shift+V) and press Enter:
   ```
   cd /workspace && wget -qO runpod_train.sh https://raw.githubusercontent.com/kempyyyy4/rupert-learning/main/cloud/runpod_train.sh && bash runpod_train.sh 100000000
   ```
10. Text will start scrolling. **You can close the tab; it keeps running.** To check on it later, reopen Jupyter; the log is the `rupert_....log` file in the file list.
11. When it finishes you'll see a big box of `#####` saying **DOWNLOAD NOW**.
12. In Jupyter's file list on the left, click the refresh icon, right-click **rupert_results_....zip**, then **Download**.
13. Back on runpod.io > **Pods**, click the **bin icon** to delete the pod. (It deletes itself after 45 minutes anyway.)
14. **Check the Pods page is empty.**
15. On your PC: right-click the zip > **Extract All…** > choose `E:\AI Channel\Rupert Sim\outputs`. Tell Claude.

## If something goes wrong
- Screenshot the terminal or the log file and send it to Claude.
- If you're unsure whether a pod is still running: **delete it.** Nothing important lives only on the pod once you've downloaded the zip.
