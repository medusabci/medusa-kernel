import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def notify_me(mailto, subject, body, mailfrom, password, host="smtp.live.com"):
    """ This method implements sending an email (ONLY TESTED FROM A HOTMAIL
    ADDRESS) to any email address. To allow the email sending from you Hotmail
    account, you have to validate  it, if you get an error please check your
    inbox (probably, a confirmation mail has been sent).

    Parameters
    ----------
    mailto : string
        Mail recipient address.
    subject : string
        Mail subject.
    body : string
        Mail body.
    mailfrom : string
        Mail sender address.
    password : string
        Password from mailfrom address.
    host : string
        SMTP host of the "mailfrom" address. In case of Hotmail use
        "smtp.live.com"

    """
    # Error check
    if type(subject) != str:
        raise ValueError("Parameter subject must be of type str")
    if type(body) != str:
        raise ValueError("Parameter body must be of type str")
    if type(mailto) != str:
        raise ValueError("Parameter mailto must be of type str")

    # Mail Sending
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
