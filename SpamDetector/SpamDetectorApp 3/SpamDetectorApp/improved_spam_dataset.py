import pandas as pd

# Load original dataset
original = pd.read_csv("spam.csv", encoding="ISO-8859-1")[["v1", "v2"]]
original.columns = ["label", "text"]

# Add realistic spam examples
custom_spam = [
    "Your account has been suspended. Please verify your credentials to restore access.",
    "Congratulations! You’ve won ₹1,00,000. Claim your reward by providing bank details.",
    "Work from home and earn ₹5000/week. No experience required. Apply now.",
    "Update your billing information to avoid service interruption.",
    "Take this 30-second survey and get a ₹500 Amazon voucher. Limited time only.",
    "Download invoice_8327.zip to review your pending payment.",
    "Complete onboarding form and confirm availability for remote position.",
    "Microsoft detected malware on your device. Call 1800-TECH-HELP to secure it.",
    "Vendor payment required before 2 PM. Use alternate account ending in 8372.",
    "Immediate transfer needed. CFO has approved confidential payment.",
    "Transfer ₹92,850 to the backup operations account ending in 4921 using temporary credentials.",
    "As per audit compliance, initiate urgent funds clearance to our secure vault via interim account.",
    "Update payroll by EOD to ensure contractor invoice is released — authorization ID: PAY6723."
]

# Add normal ham (not spam) messages
custom_ham = [
    "Hey, are we still on for the meeting at 3 PM today?",
    "Can you share the project files by tomorrow morning?",
    "Let's catch up this weekend if you're free.",
    "I’ve attached the photos from the team outing.",
    "Reminder: HR training scheduled for next Monday.",
    "Don’t forget to submit the weekly timesheet.",
    "Hey! Just checking in on how you're doing.",
    "Please review the attached slides before our Monday pitch.",
    "I’m updating the deployment notes — will send shortly.",
    "Don’t worry about the budget note, I’ll discuss with Rahul."
]

# Convert to DataFrames
extra_spam = pd.DataFrame({"label": "spam", "text": custom_spam})
extra_ham = pd.DataFrame({"label": "ham", "text": custom_ham})

# Combine original with new data
augmented = pd.concat([original, extra_spam, extra_ham], ignore_index=True)

# Save the improved dataset
augmented.to_csv("spam_improved.csv", index=False)
print("✅ Improved dataset saved as spam_improved.csv with", len(augmented), "rows.")
