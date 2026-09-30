import streamlit as st
from detectors import analyze

st.set_page_config(page_title="AI Content Checker", page_icon="🔎", layout="centered")

st.title("Kya ye content AI se bana hai?")
st.caption("Photo, video, PDF ya Word file upload karo. Hum metadata aur content dekh kar andaza lagate hain.")

f = st.file_uploader(
    "File chunein",
    type=["jpg", "jpeg", "png", "webp", "gif", "bmp", "mp4", "mov", "webm", "pdf", "docx", "txt", "md"],
)

if f:
    if f.type and f.type.startswith("image/"):
        st.image(f, width="stretch")
    elif f.type and f.type.startswith("video/"):
        st.video(f)

    with st.spinner("Jaanch ho rahi hai…"):
        try:
            res = analyze(f.name, f.getvalue())
        except ValueError as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"File process nahi ho payi: {e}"); st.stop()

    s = res["score"]
    if s is None:
        st.warning("Nateeja tay nahi ho saka")
    else:
        if s >= 75:
            st.error(f"AI hone ke majboot signal mile — andaza {s}%")
        elif s >= 40:
            st.warning(f"Kuch signal mile, pakka nahi — andaza {s}%")
        else:
            st.success(f"Koi khas AI signal nahi mila — andaza {s}%")
        st.progress(s / 100)

    st.subheader("Kaaran")
    for r in res["reasons"]:
        st.markdown(f"- {r}")
    st.caption(f"Tareeka: {res['method']}")

st.divider()
st.caption("Ye sirf andaza hai, saboot nahi. Metadata hat sakta hai aur detectors galat ho sakte hain. "
           "Is result se kisi par ilzaam mat lagao.")
