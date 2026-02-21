import streamlit as st
import os, base64
import pandas as pd
import numpy as np
import cv2
from datetime import date

# ===== Face Recognition Imports =====
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# ===== Email Sending Imports =====
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, time

def get_status_by_time():
    late_cutoff = time(8, 30)   # 08:30 AM
    current_time = datetime.now().time()

    if current_time > late_cutoff:
        return "Late"
    else:
        return "Present"



# ===================== CONFIG =====================
st.set_page_config(
    page_title="Smart Attendance - Valliammal College",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== FILE PATHS =====================
WELCOME_BG = "assets/welcome_bg.png"
LOGIN_BG = "assets/login_bg.png"
DASHBOARD_BG = "assets/dashboard_bg.png"
COLLEGE_LOGO = "assets/college_logo.png"

ATTENDANCE_FILE = "attendance.xlsx"
DATASET_FOLDER = "dataset"
EMBEDDINGS_FOLDER = "embeddings"
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

PARENTS_FILE = "parents.csv"
STAFF_FILE = "staff_users.csv"


# ===================== ✅ EMAIL CONFIG =====================
SENDER_EMAIL = "rajeswariisaravanan@gmail.com"
SENDER_APP_PASSWORD = "enkckgdxmlwcrzwz"


# ===================== BACKGROUND FUNCTION =====================
def set_bg(image_path: str):
    if not os.path.exists(image_path):
        return
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/{ext};base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ===================== GLOBAL CSS =====================
def load_css():
    st.markdown("""
    <style>

    /* ✅ Hide Streamlit UI */
    #MainMenu, footer {visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"] {display:none;}

    /* ✅ REMOVE TOP SPACE BOX */
    header[data-testid="stHeader"] {display:none !important;}
    div[data-testid="stToolbar"] {display:none !important;}
    div[data-testid="stDecoration"] {display:none !important;}

    .block-container{
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
    }

    /* ✅ REMOVE EXTRA EMPTY LABEL SPACE */
    div[data-testid="stWidgetLabel"]{
        display:none !important;
        height:0px !important;
        margin:0px !important;
        padding:0px !important;
    }

    /* ✅ Inputs */
    input {
        background: rgba(0,0,0,0.40) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: white !important;
        border-radius: 18px !important;
        padding: 14px !important;
        font-size: 15px !important;
        outline: none !important;
    }

    /* ✅ Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
        border: none !important;
        color: white !important;
        font-size: 15px !important;
        padding: 13px 18px !important;
        border-radius: 999px !important;
        font-weight: 900 !important;
        width: 100% !important;
        transition: 0.2s ease;
        box-shadow: 0 14px 34px rgba(124,58,237,0.28) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 40px rgba(124,58,237,0.40) !important;
    }

    /* ✅ Glass Card */
    .glass-card{
        background: rgba(12, 16, 26, 0.65);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 26px;
        padding: 26px;
        backdrop-filter: blur(16px);
        box-shadow: 0 18px 52px rgba(0,0,0,0.45);
    }

    
    .hero-title{
        font-size: 56px;
        font-weight: 950;
        color: blue;
        margin: 0;
        text-align:center;
        text-shadow: 0px 10px 30px rgba(0,0,0,0.35);
        letter-spacing: 0.6px;
    }
    .hero-sub{
        font-size: 15px;
        margin-top: 10px;
        color:#020617;
        text-align:center;
        line-height: 1.6;
    }

    .badge{
        display:inline-flex;
        align-items:center;
        gap:10px;
        padding: 10px 18px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.16);
        color:#020617;
        font-weight: 800;
        font-size: 13px;
        margin: 0 auto;
    }

    .feature-grid{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 22px;
    }
    .feature-card{
        background: rgba(0,0,0,0.34);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
        min-height: 90px;
    }
    .feature-title{
        color: white;
        font-size: 15px;
        font-weight: 900;
        margin: 0;
    }
    .feature-desc{
        color: rgba(255,255,255,0.72);
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.45;
    }

    @media(max-width: 900px){
        .feature-grid{ grid-template-columns: 1fr; }
        .hero-title{ font-size: 40px; }
    }
    </style>
    """, unsafe_allow_html=True)


# ===================== SESSION INIT =====================
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "menu_open" not in st.session_state:
        st.session_state.menu_open = True
    if "role" not in st.session_state:
        st.session_state.role = None



# ===================== ATTENDANCE HELPERS =====================
def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Date", "Name", "Status"])
        df.to_excel(ATTENDANCE_FILE, index=False)


def read_attendance():
    ensure_attendance_file()
    df = pd.read_excel(ATTENDANCE_FILE)

    if "Date" not in df.columns:
        df["Date"] = ""
    if "Name" not in df.columns:
        df["Name"] = ""
    if "Status" not in df.columns:
        df["Status"] = ""

    return df


def save_attendance(df):
    df.to_excel(ATTENDANCE_FILE, index=False)


def get_student_names():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    names = []
    for item in os.listdir(DATASET_FOLDER):
        path = os.path.join(DATASET_FOLDER, item)
        if os.path.isdir(path):
            names.append(item.lower())
    names.sort()
    return names


def mark_attendance(name, status="Present"):
    df = read_attendance()
    today_str = date.today().strftime("%Y-%m-%d")
    name = name.lower().strip()

    df = df[~((df["Date"].astype(str) == today_str) & (df["Name"].str.lower() == name))]

    new_row = pd.DataFrame([{"Date": today_str, "Name": name, "Status": status}])
    df = pd.concat([df, new_row], ignore_index=True)[["Date", "Name", "Status"]]
    save_attendance(df)


def attendance_summary():
    df = read_attendance()
    today_str = date.today().strftime("%Y-%m-%d")

    total_students = len(get_student_names())
    total_working_days = df["Date"].nunique() if len(df) > 0 else 0

    today_present = len(
        df[(df["Date"] == today_str) & (df["Status"] == "Present")]
    )

    today_late = len(
        df[(df["Date"] == today_str) & (df["Status"] == "Late")]
    )

    today_absent = len(
        df[(df["Date"] == today_str) & (df["Status"] == "Absent")]
    )

    return total_students, total_working_days, today_present, today_late, today_absent


def ensure_staff_file():
    if not os.path.exists(STAFF_FILE):
        pd.DataFrame(columns=["username", "password"]).to_csv(STAFF_FILE, index=False)

def save_staff(username, password):
    ensure_staff_file()
    df = pd.read_csv(STAFF_FILE)

    if username.lower() in df["username"].str.lower().tolist():
        return False

    df = pd.concat(
        [df, pd.DataFrame([{"username": username, "password": password}])],
        ignore_index=True
    )
    df.to_csv(STAFF_FILE, index=False)
    return True

def validate_staff(username, password):
    ensure_staff_file()
    df = pd.read_csv(STAFF_FILE)

    return not df[
        (df["username"].str.lower() == username.lower()) &
        (df["password"] == password)
    ].empty

def student_attendance_stats(student_name):
    df = read_attendance().copy()
    df["Name"] = df["Name"].str.lower()
    df = df[df["Name"] == student_name.lower()]

    if df.empty:
        return None, None

    status_counts = df["Status"].value_counts().reindex(
        ["Present", "Late", "Absent"], fill_value=0
    )

    df["Date"] = pd.to_datetime(df["Date"])
    daily_present = (
        df[df["Status"] == "Present"]
        .groupby("Date")
        .size()
        .reset_index(name="Present_Count")
    )

    return status_counts, daily_present



# ===================== PARENTS EMAIL HELPERS =====================
def ensure_parents_file():
    if not os.path.exists(PARENTS_FILE):
        pd.DataFrame(columns=["Name", "Parent_Email"]).to_csv(PARENTS_FILE, index=False)


def read_parents():
    ensure_parents_file()
    return pd.read_csv(PARENTS_FILE)


def save_parent_email(student_name, parent_email):
    df = read_parents()
    if parent_email.strip() == "":
        parent_email = "Unknown"
    df = df[df["Name"].str.lower() != student_name.lower()]
    df = pd.concat([df, pd.DataFrame([{"Name": student_name.lower(), "Parent_Email": parent_email}])],
                   ignore_index=True)
    df.to_csv(PARENTS_FILE, index=False)


def auto_add_unknown_for_all_students():
    students = get_student_names()
    df = read_parents()
    existing = df["Name"].str.lower().tolist() if len(df) else []
    for s in students:
        if s.lower() not in existing:
            save_parent_email(s, "Unknown")


# ===================== ✅ EMAIL SENDER =====================
def send_email_logic(parent_email, student_name, is_verification=False):
    if is_verification:
        subject = "Verification - Smart Attendance System"
        body = f"This is a test email to verify that {parent_email} is correctly linked to {student_name.upper()}."
    else:
        subject = "Attendance Alert - Absent Today"
        body = f"Hello Parent,\n\nYour ward {student_name.upper()} is marked as ABSENT today ({date.today()})."

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = parent_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=25)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, parent_email, msg.as_string())
        server.quit()
        return True, f"✅ Success"
    except Exception as e:
        return False, f"❌ Error: {e}"


# ===================== FACE MODELS =====================
@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    return detector, embedder


def extract_face(image_bgr, box, size=(160, 160)):
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    face = image_bgr[y:y + h, x:x + w]
    if face.size == 0:
        return None
    return cv2.resize(face, size)


def get_embedding(embedder, face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return embedder.embeddings(face_rgb)[0]


def save_student_embedding(student_name, embedding):
    np.save(os.path.join(EMBEDDINGS_FOLDER, f"{student_name.lower()}.npy"), embedding)


def load_all_embeddings():
    known = {}
    for file in os.listdir(EMBEDDINGS_FOLDER):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            known[name.lower()] = np.load(os.path.join(EMBEDDINGS_FOLDER, file), allow_pickle=True)
    return known


def build_embeddings_from_dataset():
    detector, embedder = load_models()
    students = get_student_names()

    if not students:
        return False, "⚠️ No students in dataset/"

    built = 0
    skipped = 0
    failed_students = []

    progress = st.progress(0)
    status_text = st.empty()

    for idx, student in enumerate(students):
        folder = os.path.join(DATASET_FOLDER, student)
        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

        if not imgs:
            skipped += 1
            failed_students.append(f"{student} (no images)")
            continue

        found_face = False

        for img_file in imgs:
            img_path = os.path.join(folder, img_file)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                continue

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            if not faces:
                continue

            faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)
            face = extract_face(img_bgr, faces[0]["box"])

            if face is None:
                continue

            emb = get_embedding(embedder, face)
            save_student_embedding(student.lower(), emb)

            built += 1
            found_face = True
            break

        if not found_face:
            skipped += 1
            failed_students.append(f"{student} (no face detected)")

        progress.progress((idx + 1) / len(students))
        status_text.info(f"Processing: {student} ({idx + 1}/{len(students)})")

    if built == 0:
        return False, f"❌ No embeddings created!\n\nFailed: {failed_students}"

    msg = f"✅ Embeddings created for {built} students | Skipped: {skipped}"
    if failed_students:
        msg += "\n\n⚠️ Failed Students:\n" + "\n".join(failed_students)

    return True, msg


def recognize_faces_in_image(image_bgr, threshold=0.45):
    detector, embedder = load_models()
    known = load_all_embeddings()

    if not known:
        return [], "⚠️ No embeddings found! Generate embeddings first."

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        return [], "❌ No faces detected."

    results = []
    for f in faces:
        box = f["box"]
        face = extract_face(image_bgr, box)
        if face is None:
            continue

        emb = get_embedding(embedder, face)

        best_name = "Unknown"
        best_dist = 999

        for name, known_emb in known.items():
            dist = cosine(emb, known_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist > threshold:
            best_name = "Unknown"

        results.append({"name": best_name, "box": box, "dist": round(best_dist, 3)})

    return results, "✅ Recognition completed!"


def draw_boxes(image_bgr, detections):
    img = image_bgr.copy()

    for d in detections:
        x, y, w, h = d["box"]
        name = d["name"]

        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        label = str(name).upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(0, y - th - 12)

        cv2.rectangle(img, (x, label_y), (x + tw + 12, label_y + th + 10), (0, 0, 0), -1)
        cv2.putText(img, label, (x + 6, label_y + th + 3), font, font_scale, (255, 255, 255), thickness)

    return img


# ===================== MENU =====================
def menu_toggle_button():
    if st.button("☰ Menu", key="toggle_menu_btn"):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()


def left_menu():
    role = st.session_state.get("role", "staff")  # default = staff

    if st.button("🏠 Home", key="m_home"):
        st.session_state.page = "home"; st.rerun()

    if st.button("📊 Dashboard", key="m_dash"):
        st.session_state.page = "dashboard"; st.rerun()

    if st.button("✅ Attendance", key="m_att"):
        st.session_state.page = "mark"; st.rerun()

    if st.button("📄 View Records", key="m_view"):
        st.session_state.page = "view"; st.rerun()

    if st.button("📈 Attendance %", key="m_per"):
        st.session_state.page = "percentage"; st.rerun()

    # ✅ STUDENT PROFILE (VISIBLE FOR ALL)
    if st.button("🧑‍🎓 Student Profile", key="m_student"):
        st.session_state.page = "student_profile"
        st.rerun()

    # 🔒 ADMIN ONLY OPTIONS
    if role == "admin":
        if st.button("📁 Dataset", key="m_data"):
            st.session_state.page = "dataset"; st.rerun()

        if st.button("📧 Parent Email", key="m_email"):
            st.session_state.page = "email"; st.rerun()

        if st.button("⬇️ Excel Download", key="m_xl"):
            st.session_state.page = "excel"; st.rerun()

    # 🚪 LOGOUT (ALWAYS LAST)
    if st.button("🚪 Logout", key="m_logout"):
        st.session_state.logged_in = False
        st.session_state.page = "welcome"
        st.rerun()


def layout_with_menu(render_function):
    set_bg(DASHBOARD_BG)
    load_css()

    col_menu, col_main = st.columns([1, 4])

    with col_menu:
        menu_toggle_button()
        if st.session_state.menu_open:
            left_menu()

    with col_main:
        render_function()


# ===================== ✅ PROFESSIONAL WEBSITE HOME PAGE =====================
def welcome_page():
    set_bg(WELCOME_BG)
    load_css()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Main centered layout
    colL, colC, colR = st.columns([1, 2.4, 1])

    with colC:
        if os.path.exists(COLLEGE_LOGO):
            st.image(COLLEGE_LOGO, width=820)

        # Title section
        st.markdown("""
            <div style="text-align:center; margin-top:18px;">
                <div class="badge">✅ AI Powered · Face Recognition · Attendance Automation</div>
                <div style="margin-top:18px;"></div>
                <h1 class="hero-title">Smart Attendance</h1>
                <p class="hero-sub">
                    Modern attendance solution using Face Recognition<br>
                    for Valliammal College for Women
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("""
            <div class="feature-grid">
                <div class="feature-card">
                    <p class="feature-title">⚡ Fast Attendance</p>
                    <p class="feature-desc">Mark attendance from group photo in seconds.</p>
                </div>
                <div class="feature-card">
                    <p class="feature-title">📊 Analytics Dashboard</p>
                    <p class="feature-desc">Working days, present & absent insights easily.</p>
                </div>
                <div class="feature-card">
                    <p class="feature-title">📧 Parent Alerts</p>
                    <p class="feature-desc">Send absence updates to parents securely.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

        # ✅ SIDE-BY-SIDE BUTTONS (ONLY ONE COLUMNS CALL)
        btn1, btn2 = st.columns(2, gap="large")

        with btn1:
            if st.button("🔑 Sign in", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()

        with btn2:
            if st.button("📝 Sign up", use_container_width=True):
                st.session_state.page = "register"
                st.rerun()

        # Footer
        st.markdown("""
            <p style="color:rgba(255,255,255,0.45);
                      text-align:center;
                      margin-top:35px;
                      font-size:12px;">
                © Smart Attendance System · Valliammal College for Women
            </p>
        """, unsafe_allow_html=True)


def register_page():
    set_bg(LOGIN_BG)
    load_css()

    st.markdown("<br><br>", unsafe_allow_html=True)

    colL, colC, colR = st.columns([1, 2, 1])

    with colC:
        if os.path.exists(COLLEGE_LOGO):
            st.image(COLLEGE_LOGO, width=700)

        st.markdown(
            "<h2 style='text-align:center; color:white;'>Staff Registration</h2>",
            unsafe_allow_html=True
        )

        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")
        confirm_p = st.text_input("Confirm Password", type="password")

        if st.button("✅ Register"):
            if not new_u or not new_p:
                st.error("Fields cannot be empty!")

            elif new_p != confirm_p:
                st.error("Passwords do not match!")

            else:
                success = save_staff(new_u.strip(), new_p.strip())
                if success:
                    st.success("✅ Staff registered successfully! Please login.")
                else:
                    st.error("⚠️ Username already exists")

        if st.button("⬅️ Back to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()




# ===================== ✅ PROFESSIONAL LOGIN PAGE =====================
def login_page():
    set_bg(LOGIN_BG)
    load_css()

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

    colL, colC, colR = st.columns([1.2, 2.2, 1.2])
    with colC:
        if os.path.exists(COLLEGE_LOGO):
            st.image(COLLEGE_LOGO, width=780)

        st.markdown("""
            <div style="text-align:center; margin-top:18px;">
                <h2 style="color:white; font-weight:950; margin:0; font-size:44px;">
                    Login
                </h2>
                <p style="color:rgba(255,255,255,0.80); margin-top:10px; font-size:15px;">
                    Admin / Staff Login
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
        password = st.text_input("Password", placeholder="Password", type="password", label_visibility="collapsed")

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        if st.button("🔐 Login"):
            username = username.strip()
            password = password.strip()

            # -------- ADMIN LOGIN --------
            if username == "raje" and password == "raje@1234":
                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.session_state.page = "home"
                st.success("✅ Admin Login Successful")
                st.rerun()

            # -------- STAFF LOGIN --------
            elif validate_staff(username, password):
                st.session_state.logged_in = True
                st.session_state.role = "staff"
                st.session_state.page = "home"
                st.success("✅ Staff Login Successful")
                st.rerun()

            else:
                st.error("❌ Invalid username or password")

        if st.button("⬅️ Back"):
            st.session_state.page = "welcome"
            st.rerun()



# ===================== PAGES =====================
def home_page():
    def content():
        role = st.session_state.role or "staff"

        if role == "admin":
            title = "👋 Welcome, Admin"
            badge = "🛡️ ADMIN ACCESS"
            lines = [
                "🤖 AI-powered face recognition attendance",
                "📊 Student datasets, reports & dashboards",
                "📧 Automated parent notifications"
            ]
        else:
            title = "👋 Welcome, Staff"
            badge = "👨‍🏫 STAFF ACCESS"
            lines = [
                "📸 Mark attendance using face recognition",
                "📄 View daily & monthly attendance records",
                "📈 Monitor student attendance easily"
            ]

        st.markdown(
f"""
<div class="glass-card">

  <div style="display:flex; justify-content:space-between; align-items:center;">
    <h2 style="color:white; margin:0;">{title}</h2>
    <span style="
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 14px;
        border-radius: 999px;
        color: white;
        font-size: 12px;
        font-weight: 800;
    ">
      {badge}
    </span>
  </div>

  <p style="color:rgba(255,255,255,0.85); margin-top:12px; font-size:15px;">
    You are logged in to the <b>Smart Attendance System</b>.
  </p>

  {''.join([
      f"<p style='color:rgba(255,255,255,0.7); font-size:13px; margin-top:6px;'>{line}</p>"
      for line in lines
  ])}

  <p style="color:rgba(255,255,255,0.6); margin-top:12px; font-size:13px;">
    Smart • Secure • Accurate Attendance Management
  </p>

  
</div>
""".lstrip(),
            unsafe_allow_html=True
        )

    layout_with_menu(content)

def is_attendance_marked_today():
    df = read_attendance()
    today_str = date.today().strftime("%Y-%m-%d")
    return not df[df["Date"] == today_str].empty



def dashboard_page():
    def content():
        st.markdown("## 📊 Dashboard Overview")

        # ================= TODAY STATUS BANNER =================
        today_marked = is_attendance_marked_today()
        today_str = date.today().strftime("%d %b %Y")

        if today_marked:
            st.markdown(f"""
            <div class="glass-card" style="border-left:6px solid #22c55e; margin-bottom:20px;">
                <h3 style="color:#22c55e; margin:0;">
                    ✅ Attendance Marked for Today
                </h3>
                <p style="color:rgba(255,255,255,0.75); margin-top:6px;">
                    Date: {today_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card" style="border-left:6px solid #facc15; margin-bottom:20px;">
                <h3 style="color:#facc15; margin:0;">
                    ⚠️ Attendance NOT Marked Yet
                </h3>
                <p style="color:rgba(255,255,255,0.75); margin-top:6px;">
                    Please mark attendance for today ({today_str})
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ================= KPI DATA =================
        (
            total_students,
            total_working_days,
            today_present,
            today_late,
            today_absent
        ) = attendance_summary()

        a, b, c, d, e = st.columns(5)

        with a:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#38bdf8;">Total Students</h4>
                <h1 style="color:#38bdf8;">{total_students}</h1>
            </div>
            """, unsafe_allow_html=True)

        with b:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#e5e7eb;">Working Days</h4>
                <h1 style="color:#e5e7eb;">{total_working_days}</h1>
            </div>
            """, unsafe_allow_html=True)

        with c:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#4ade80;">Today Present</h4>
                <h1 style="color:#4ade80;">{today_present}</h1>
            </div>
            """, unsafe_allow_html=True)

        with d:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#facc15;">Today Late</h4>
                <h1 style="color:#facc15;">{today_late}</h1>
            </div>
            """, unsafe_allow_html=True)

        with e:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#f87171;">Today Absent</h4>
                <h1 style="color:#f87171;">{today_absent}</h1>
            </div>
            """, unsafe_allow_html=True)

    layout_with_menu(content)


def mark_page():
    def content():
        st.markdown("## ✅ Attendance Marking")

        uploaded_files = st.file_uploader(
            "📂 Upload Images (Multiple Allowed)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True
        )

        if uploaded_files:
            threshold = st.slider("Face Threshold", 0.30, 0.70, 0.45, 0.01)

            if st.button("🔍 Detect & Mark Attendance", key="detect_btn"):
                all_present = set()

                for uploaded in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img_bgr is None:
                        continue

                    detections, msg = recognize_faces_in_image(
                        img_bgr, threshold=threshold
                    )

                    for d in detections:
                        if d["name"] != "Unknown":
                            auto_status = get_status_by_time()
                            mark_attendance(d["name"], auto_status)
                            all_present.add(d["name"])

                    boxed = draw_boxes(img_bgr, detections)
                    st.image(
                        cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                        caption=uploaded.name,
                        width=900
                    )

                st.success("✅ Attendance marked from all images")

                st.markdown("### ✅ Present Students (From All Images)")
                if all_present:
                    st.dataframe(
                        pd.DataFrame({"Name": sorted(all_present)}),
                        use_container_width=True
                    )
                else:
                    st.info("No recognized students")

        # ---------------- MANUAL ATTENDANCE ----------------
        st.markdown("---")
        st.markdown("### ✍️ Manual Attendance")

        students = get_student_names()
        if students:
            c1, c2 = st.columns(2)
            with c1:
                selected = st.selectbox("Student Name", students)
            with c2:
                status = st.radio(
                    "Status", ["Present", "Late", "Absent"], horizontal=True
                )

            if st.button("✅ Save Manual Attendance", key="manual_save"):
                mark_attendance(selected, status)
                st.success(f"✅ Marked {selected} as {status}")
        else:
            st.warning("⚠️ No students in dataset folder.")

        # ---------------- EMAIL NOTIFICATION ----------------
        st.markdown("---")
        st.markdown("### 📧 Parent Notifications")

        if st.button("✉️ Send Absentee Alerts (Today)", key="send_bulk_email"):
            today_str = date.today().strftime("%Y-%m-%d")
            df_att = read_attendance()
            df_parents = read_parents()

            absent_today = df_att[
                (df_att["Date"] == today_str) &
                (df_att["Status"] == "Absent")
            ]

            if absent_today.empty:
                st.info("No students marked as 'Absent' today.")
            else:
                with st.spinner("Sending emails..."):
                    for _, row in absent_today.iterrows():
                        s_name = row["Name"].lower()
                        p_row = df_parents[
                            df_parents["Name"].str.lower() == s_name
                        ]

                        if not p_row.empty:
                            p_email = p_row.iloc[0]["Parent_Email"]

                            if (
                                isinstance(p_email, str)
                                and "@" in p_email
                                and p_email.lower() != SENDER_EMAIL.lower()
                            ):
                                success, res_msg = send_email_logic(
                                    p_email, s_name
                                )
                                if success:
                                    st.write(f"✅ Sent: {s_name}")
                                else:
                                    st.write(f"❌ Failed: {s_name} ({res_msg})")

                st.success("✅ Email process completed")

    layout_with_menu(content)

def student_profile_page():
    def content():
        st.markdown("## 🧑‍🎓 Student Attendance Profile")

        students = get_student_names()
        if not students:
            st.warning("⚠️ No students found in dataset.")
            return

        # ---- Student Selector ----
        student = st.selectbox("Select Student", students)

        df = read_attendance().copy()
        if df.empty:
            st.info("No attendance data available.")
            return

        df["Name"] = df["Name"].str.lower()
        student_df = df[df["Name"] == student.lower()]

        if student_df.empty:
            st.warning("No records found for this student.")
            return

        # ---- Metrics ----
        total_days = df["Date"].nunique()
        present_count = len(student_df[student_df["Status"] == "Present"])
        late_count = len(student_df[student_df["Status"] == "Late"])
        absent_count = len(student_df[student_df["Status"] == "Absent"])

        attendance_percent = round((present_count / total_days) * 100, 2) if total_days else 0

        # ---- KPI Cards ----
        a, b, c, d, e = st.columns(5)

        with a:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#38bdf8;">Total Days</h4>
                <h1 style="color:#38bdf8;">{total_days}</h1>
            </div>
            """, unsafe_allow_html=True)

        with b:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#4ade80;">Present</h4>
                <h1 style="color:#4ade80;">{present_count}</h1>
            </div>
            """, unsafe_allow_html=True)

        with c:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#facc15;">Late</h4>
                <h1 style="color:#facc15;">{late_count}</h1>
            </div>
            """, unsafe_allow_html=True)

        with d:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:#f87171;">Absent</h4>
                <h1 style="color:#f87171;">{absent_count}</h1>
            </div>
            """, unsafe_allow_html=True)

        with e:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color:white;">Attendance %</h4>
                <h1 style="color:white;">{attendance_percent}%</h1>
            </div>
            """, unsafe_allow_html=True)

        # ================== 📊 ATTENDANCE GRAPHS ==================
        st.markdown("### 📊 Attendance Analytics")

        status_counts, daily_present = student_attendance_stats(student)

        if status_counts is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Status Distribution")
                st.bar_chart(status_counts)

            with col2:
                if daily_present is not None and not daily_present.empty:
                    st.markdown("#### Attendance Trend")
                    st.line_chart(
                        daily_present.set_index("Date")["Present_Count"]
                    )
        else:
            st.info("No data available for charts.")

        # ---- Attendance History ----
        st.markdown("---")
        st.markdown("### 📄 Attendance History")

        student_df = student_df.sort_values("Date", ascending=False)
        st.dataframe(
            student_df[["Date", "Status"]],
            use_container_width=True
        )

    layout_with_menu(content)



def view_page():
    def content():
        st.markdown("## 📄 Attendance Records (Month Wise)")

        df = read_attendance().copy()
        if df.empty:
            st.info("No attendance record found.")
            return

        # Convert Date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Month"] = df["Date"].dt.strftime("%Y-%m")

        # ---- Month selector ----
        months = sorted(df["Month"].unique(), reverse=True)
        selected_month = st.selectbox("📅 Select Month", months)

        month_df = df[df["Month"] == selected_month]
        if month_df.empty:
            st.warning("No data for selected month.")
            return

        # ---- Date selector ----
        dates = sorted(
            month_df["Date"].dt.strftime("%Y-%m-%d").unique(),
            reverse=True
        )
        selected_date = st.selectbox("📆 Select Date", dates)

        day_df = month_df[
            month_df["Date"].dt.strftime("%Y-%m-%d") == selected_date
        ]

        # ---- Split by status ----
        present_df = day_df[day_df["Status"] == "Present"][["Name", "Status"]].reset_index(drop=True)
        late_df    = day_df[day_df["Status"] == "Late"][["Name", "Status"]].reset_index(drop=True)
        absent_df  = day_df[day_df["Status"] == "Absent"][["Name", "Status"]].reset_index(drop=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"### ✅ Present ({selected_date})")
            if present_df.empty:
                st.info("None")
            else:
                st.dataframe(present_df, use_container_width=True)

        with col2:
            st.markdown(f"### 🕒 Late ({selected_date})")
            if late_df.empty:
                st.info("None")
            else:
                st.dataframe(late_df, use_container_width=True)

        with col3:
            st.markdown(f"### ❌ Absent ({selected_date})")
            if absent_df.empty:
                st.info("None")
            else:
                st.dataframe(absent_df, use_container_width=True)

    layout_with_menu(content)


def percentage_page():
    def content():
        st.markdown("## 📈 Attendance Percentage")
        df = read_attendance()
        students = get_student_names()

        if len(df) == 0 or len(students) == 0:
            st.info("Not enough data")
            return

        total_days = df["Date"].nunique()
        result = []
        for s in students:
            present_count = len(df[(df["Name"].str.lower() == s.lower()) & (df["Status"] == "Present")])
            percent = round((present_count / total_days) * 100, 2) if total_days > 0 else 0
            result.append({"Name": s, "Present Days": present_count, "Percentage": f"{percent}%"})

        st.dataframe(pd.DataFrame(result), use_container_width=True)

    layout_with_menu(content)


def dataset_page():
    def content():
        st.markdown("## 📁 Dataset Students")

        abs_path = os.path.abspath(DATASET_FOLDER)
        st.info(f"📂 **Dataset Folder Location:** `{abs_path}`")

        # ✅ OPEN FOLDER BUTTON
        if st.button("📂 Open Dataset Folder"):
            try:
                if os.name == "nt":  # Windows
                    os.startfile(abs_path)
                elif os.name == "posix":  # macOS / Linux
                    import subprocess
                    subprocess.Popen(["open", abs_path])
            except Exception as e:
                st.error(f"Unable to open folder: {e}")

        st.markdown("""
        **How to add students:**
        - Open the dataset folder
        - Create a folder with the **student name**
        - Put face images inside
        
        Example:
        ```
        dataset/
        ├── priya/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── lavanya/
        │   └── photo.png
        ```
        """)

        st.markdown("---")

        students = get_student_names()
        if not students:
            st.warning("⚠️ No students found in dataset folder.")
            return

        st.success(f"✅ Found {len(students)} students")
        st.dataframe(pd.DataFrame({"Students": students}), use_container_width=True)

        st.markdown("---")
        st.markdown("## 🧠 Generate Embeddings")

        if st.button("⚡ Generate Embeddings", key="gen_embed"):
            with st.spinner("Generating embeddings..."):
                ok, msg = build_embeddings_from_dataset()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    layout_with_menu(content)



def parent_email_page():
    def content():
        st.markdown("## 📧 Parent Email Management")

        auto_add_unknown_for_all_students()
        students = get_student_names()
        if not students:
            st.warning("⚠️ No students")
            return

        # Section 1: Update Email
        st.markdown("### ✍️ Add/Update Email")
        student = st.selectbox("Select Student", students)
        df_parents = read_parents()
        old = df_parents[df_parents["Name"].str.lower() == student.lower()]["Parent_Email"].values
        default_email = old[0] if len(old) else "Unknown"

        email_input = st.text_input("Enter Parent Email", value=str(default_email))

        if st.button("💾 Save Email Record", key="save_parent"):
            save_parent_email(student, email_input)
            st.success(f"✅ Email saved for {student}")
            st.rerun()

        st.markdown("---")
        
        # Section 2: Verification List
        st.markdown("### 🔍 Verify Email Addresses")
        st.info("Click 'Verify' to send a test email and confirm the address is working.")
        
        df_display = read_parents()
        
        # Table Header
        h1, h2, h3 = st.columns([2, 3, 1])
        h1.markdown("**Student Name**")
        h2.markdown("**Parent Email**")
        h3.markdown("**Action**")
        
        st.markdown("<hr style='margin:5px 0px;'>", unsafe_allow_html=True)

        # Table Rows with Buttons
        for index, row in df_display.iterrows():
            c1, c2, c3 = st.columns([2, 3, 1])
            c1.write(row['Name'].upper())
            c2.write(row['Parent_Email'])
            
            # Button for each row
            if c3.button("Verify", key=f"verify_{index}"):
                if row['Parent_Email'] == "Unknown" or "@" not in str(row['Parent_Email']):
                    st.error("Invalid Email")
                else:
                    with st.spinner("Sending..."):
                        success, result = send_email_logic(row['Parent_Email'], row['Name'], is_verification=True)
                        if success:
                            st.toast(f"Test email sent to {row['Name']}!", icon="✅")
                        else:
                            st.error(result)

    layout_with_menu(content)


def excel_page():
    def content():
        st.markdown("## ⬇️ Download Excel")
        ensure_attendance_file()
        with open(ATTENDANCE_FILE, "rb") as f:
            st.download_button(
                "📥 Download attendance.xlsx",
                data=f,
                file_name="attendance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    layout_with_menu(content)


# ===================== MAIN ROUTER =====================
def main():
    init_session()
    
    if not st.session_state.logged_in:
        if st.session_state.page == "welcome":
            welcome_page()
        elif st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "register":
            register_page()
        else:
            st.session_state.page = "welcome"
            st.rerun()
        return

    pages = {
    "home": home_page,
    "dashboard": dashboard_page,
    "mark": mark_page,
    "view": view_page,
    "percentage": percentage_page,
    "dataset": dataset_page,
    "email": parent_email_page,
    "excel": excel_page,
    "student_profile": student_profile_page 
}

    if st.session_state.page in pages:
        pages[st.session_state.page]()
    else:
        st.session_state.page = "home"
        st.rerun()


if __name__ == "__main__":
    main()