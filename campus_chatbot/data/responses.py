"""
Simple intent -> canned response lookup.
Used by all three methods so the demo answers look consistent.
Feel free to expand / translate these per teammate's needs.
"""

RESPONSES = {
    "exam_timetable": "You can check your exam timetable via the Student Portal > Academic > Exam Timetable.",
    "course_registration": "Course registration / add-drop can be done via the Student Portal > Course Registration during the add-drop period.",
    "fee_payment": "You can pay your fees online via the Student Portal > Finance > Fee Payment, or at the Finance counter.",
    "hostel_application": "Hostel applications are submitted via the Student Portal > Hostel > Apply for Accommodation before the semester deadline.",
    "library_service": "Library services (borrowing, renewing, online databases) can be accessed via the Library Portal or at the library counter.",
    "it_support": "For IT issues (wifi, portal login, password reset), please contact IT Support at itsupport@tarumt.edu.my or visit the IT Helpdesk.",
    "fallback": "Sorry, I'm not confident about that yet. Could you rephrase your question, or contact the relevant department directly?",
}

CONFIDENCE_THRESHOLD = 0.35  # below this, fall back to the generic response
