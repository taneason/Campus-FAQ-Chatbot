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
    "class_timetable": (
        "Your weekly class timetable is available in the Student Portal: Academic > Class Timetable. "
        "If a class or venue looks wrong, confirm your course registration is finalized and contact the Academic Office to correct the timetable."
    ),
    "student_id_card": (
        "New or replacement student ID cards are issued at the Student Affairs counter; bring your admission letter or a police report if the card is lost. "
        "There may be a small replacement fee, which you can check with the Student Affairs Office."
    ),
    "scholarship_financial_aid": (
        "Scholarship and financial aid applications are submitted through the Student Portal > Financial Aid > Scholarship Application within the announced application period. "
        "Check your eligibility requirements first, and contact the Student Financial Aid Office if you need help with documents or application status."
    ),
    "transcript_request": (
        "Official transcripts and academic statements can be requested through the Student Portal > Academic > Transcript Request, usually with a processing fee. "
        "Processing takes a few working days; contact the Academic/Examinations Office if you need it urgently or the request status is unclear."
    ),
    "counseling_service": (
        "Free and confidential counseling support is available through the Student Counseling Center; you can book an appointment via the Student Portal or walk in during office hours. "
        "If it's urgent, please contact the counseling center directly or campus security for immediate assistance."
    ),
    "parking_permit": (
        "Campus parking permits are applied for through the Student Portal > Facilities > Parking Permit, subject to available quota. "
        "Bring your vehicle registration details, and contact the Facilities/Security Office if your permit application is rejected or delayed."
    ),
    "greeting": (
        "Hello! I'm the Campus FAQ Assistant. Ask me about exams, course registration, fees, hostel, library, IT support, and more."
    ),
    "fallback": (
        "Sorry, I'm not fully confident about that yet. Please rephrase your question or contact the relevant office directly for the most accurate answer."
    ),
}

CONFIDENCE_THRESHOLD = 0.35  # below this, fall back to the generic response
