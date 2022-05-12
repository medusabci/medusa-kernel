import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def notify_me(mailto, subject, body):
    """
    This funcion sends an email FROM A HOTMAIL ADDRESS to any email address. 
    To allow the email sending from you hotmail account, you have to validate
    it, if you get an error please chech your inbox
    
    Parameters
    ----------
    subject : str
        Email subject.
    body : str
        Email body.
    mailto : str
        Receiver's email address.
    mailfrom : str
        Sender's email address. ONLY HOTMAIL SUPORTED.
    password : str
        Password of the sender's email.

    Returns
    -------
    None.

    """
    # ================================================== ERROR CHECK ================================================= #
    if type(subject) != str:
        raise ValueError("Parameter subject must be of type str")
    if type(body) != str:
        raise ValueError("Parameter body must be of type str")
    if type(mailto) != str:
        raise ValueError("Parameter mailto must be of type str")        
    
    # ============================================ VARIABLE INITIALIZATION =========================================== #
    mailfrom = "medusa_py@hotmail.com"  # Mail created ad-hoc for this purpose only
    password = "1234abcd1234abcd"       # Ultra high secure password
    host = "smtp.live.com"

    # =============================================== MAIL SENDING =================================================== #
    # SMTP connection
    server = smtplib.SMTP(host, 587)
    server.starttls()
    server.login(mailfrom, password)
    # Message creation
    msg = MIMEMultipart()
    msg['From'] = mailfrom
    msg['To'] = mailto
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    # Email sending
    server.sendmail(
      mailfrom, 
      mailto, 
      msg.as_string())
    # Connection closing
    server.quit()
