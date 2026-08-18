"""
Simple intent -> canned response lookup.
Used by all three methods so the demo answers look consistent.
Feel free to expand / translate these per teammate's needs.
"""

RESPONSES = {
    "exam_timetable": (
        "You can check your exam timetable in the Student Portal: Academic > Exam Timetable. "
        "If the timetable is not visible, verify that your course registration is completed and contact the Academic Office if the schedule still appears missing or incorrect."
    ),
    "course_registration": (
        "Course registration and add/drop are done through the Student Portal > Course Registration during the registration window. "
        "Please check the academic calendar for the exact dates, and contact your faculty advisor if you need help with subject selection or approval."
    ),
    "fee_payment": (
        "To pay fees, go to the Student Portal > Finance > Fee Payment. "
        "You may also pay at the Finance counter if needed. If you have an outstanding balance, check the payment due date and contact the Finance Office for installment or late-payment guidance."
    ),
    "hostel_application": (
        "Hostel applications are submitted through the Student Portal > Hostel > Apply for Accommodation before the application deadline. "
        "Please prepare your student details and check room availability. If your application is unsuccessful or the portal is closed, contact the Residential Services Office for the next intake."
    ),
    "library_service": (
        "Library services such as borrowing, renewing books, and accessing online databases are available through the Library Portal or at the library counter. "
        "If you cannot access a database or need help with due dates, contact the library staff directly for assistance."
    ),
    "it_support": (
        "For IT issues such as campus Wi-Fi, portal login, password reset, or email access, please contact the IT Helpdesk or email itsupport@tarumt.edu.my. "
        "Before contacting them, make sure your login credentials are correct and try reconnecting to the campus network or resetting your password via the portal."
    ),
    "fallback": (
        "Sorry, I'm not fully confident about that yet. Please rephrase your question or contact the relevant office directly for the most accurate answer."
    ),
}

CONFIDENCE_THRESHOLD = 0.35  # below this, fall back to the generic response
