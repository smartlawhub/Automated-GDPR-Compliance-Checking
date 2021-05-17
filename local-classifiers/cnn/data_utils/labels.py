# MAPPING OF ATTRIBUTE TO ATTRIBUTE VALUES

from collections import OrderedDict
import glob
import os 

def labels_dict(path, attribute, is_parent=False):
    if attribute == 'Retention Period':
        labels = OrderedDict([('Stated Period', 0),
                              ('Limited', 1),
                              ('Indefinitely', 2),
                              ('Unspecified', 3)])
    elif attribute == 'Retention Purpose':
        labels = OrderedDict([('Advertising', 0),
                              ('Analytics/Research', 1),
                              ('Legal requirement', 2),
                              ('Marketing', 3),
                              ('Perform service', 4),
                              ('Service operation and security', 5),
                              ('Unspecified', 6)])
    elif attribute == 'Notification Type':
        labels = OrderedDict([('General notice in privacy policy', 0),
                              ('General notice on website', 1),
                              ('No notification', 2),
                              ('Personal notice', 3),
                              ('Unspecified', 4)])
    elif attribute == 'Security Measure':
        labels = OrderedDict([('Generic', 0),
                              ('Data access limitation', 1),
                              ('Privacy review/audit', 2),
                              ('Privacy training', 3),
                              ('Privacy/Security program', 4),
                              ('Secure data storage', 5),
                              ('Secure data transfer', 6),
                              ('Secure user authentication', 7),
                              ('Unspecified', 8)])
    elif attribute == 'Audience Type':
        labels = OrderedDict([('Children', 0),
                              ('Californians', 1),
                              ('Citizens from other countries', 2),
                              ('Europeans', 3)])
    elif attribute == 'User Type':
        labels = OrderedDict([('User with account', 0),
                              ('User without account', 1),
                              ('Unspecified', 2)])
    elif attribute == 'Access Scope':
        labels = OrderedDict([('Profile data', 0),
                              ('Transactional data', 1),
                              ('User account data', 2),
                              ('Other data about user', 3),
                              ('Unspecified', 4)])
    elif attribute == 'Does or Does Not':
        labels = OrderedDict([('Does', 0),
                              ('Does Not', 1)])
    elif attribute == 'Access Type':
        labels = OrderedDict([('Deactivate account', 0),
                              ('Delete account (full)', 1),
                              ('Delete account (partial)', 2),
                              ('Edit information', 3),
                              ('View', 4),
                              ('None', 5),
                              ('Unspecified', 6)])
    elif attribute == 'Action First-Party':
        labels = OrderedDict([('Collect from user on other websites', 0),
                              ('Collect in mobile app', 1),
                              ('Collect on mobile website', 2),
                              ('Collect on website', 3),
                              ('Receive from other parts of company/affiliates',
                               4),
                              ('Receive from other service/third-party (named)',
                               5),
                              (
                                  'Receive from other service/third-party (unnamed)',
                                  6),
                              ('Track user on other websites', 7),
                              ('Unspecified', 8)])
    elif attribute == 'Action Third-Party':
        labels = OrderedDict([('Collect on first party website/app', 0),
                              ('Receive/Shared with', 1),
                              ('See', 2),
                              ('Track on first party website/app', 3),
                              ('Unspecified', 4)])
    elif attribute == 'Third Party Entity':
        labels = OrderedDict([('Named third party', 0),
                              ('Other part of company/affiliate', 1),
                              ('Other users', 2),
                              ('Public', 3),
                              ('Unnamed third party', 4),
                              ('Unspecified', 5)])
    elif attribute == 'Choice Scope':
        labels = OrderedDict([('Collection', 0),
                              ('First party collection', 1),
                              ('First party use', 2),
                              ('Third party sharing/collection', 3),
                              ('Third party use', 4),
                              ('Both', 5),
                              ('Use', 6),
                              ('Unspecified', 7)])
    elif attribute == 'Choice Type':
        labels = OrderedDict([('Browser/device privacy controls', 0),
                              ('Dont use service/feature', 1),
                              ('First-party privacy controls', 2),
                              ('Opt-in', 3),
                              ('Opt-out link', 4),
                              ('Opt-out via contacting company', 5),
                              ('Third-party privacy controls', 6),
                              ('Unspecified', 7)])
    elif attribute == 'User Choice':
        labels = OrderedDict([('None', 0),
                              ('Opt-in', 1),
                              ('Opt-out', 2),
                              ('User participation', 3),
                              ('Unspecified', 4)])
    elif attribute == 'Change Type':
        labels = OrderedDict([('In case of merger or acquisition', 0),
                              ('Non-privacy relevant change', 1),
                              ('Privacy relevant change', 2),
                              ('Unspecified', 3)])
    elif attribute == 'Collection Mode':
        labels = OrderedDict([('Explicit', 0),
                              ('Implicit', 1),
                              ('Unspecified', 2)])
    elif attribute == 'Identifiability':
        labels = OrderedDict([('Aggregated or anonymized', 0),
                              ('Identifiable', 1),
                              ('Unspecified', 2)])
    elif attribute == 'Personal Information Type':
        labels = OrderedDict([('Computer information', 0),
                              ('Contact', 1),
                              ('Cookies and tracking elements', 2),
                              ('Demographic', 3),
                              ('Financial', 4),
                              ('Generic personal information', 5),
                              ('Health', 6),
                              ('IP address and device IDs', 7),
                              ('Location', 8),
                              ('Personal identifier', 9),
                              ('Social media data', 10),
                              ('Survey data', 11),
                              ('User online activities', 12),
                              ('User profile', 13),
                              ('Unspecified', 14)])
    elif attribute == 'Purpose':
        labels = OrderedDict([('Additional service/feature', 0),
                              ('Advertising', 1),
                              ('Analytics/Research', 2),
                              ('Basic service/feature', 3),
                              ('Legal requirement', 4),
                              ('Marketing', 5),
                              ('Merger/Acquisition', 6),
                              ('Personalization/Customization', 7),
                              ('Service operation and security', 8),
                              ('Unspecified', 9)])
    elif attribute == 'Majority':
        labels = OrderedDict([('First Party Collection/Use', 0),
                     ('Third Party Sharing/Collection', 1),
                     ('User Access, Edit and Deletion', 2),
                     ('Data Retention', 3),
                     ('Data Security', 4),
                     ('International and Specific Audiences', 5),
                     ('Do Not Track', 6),
                     ('Policy Change', 7),
                     ('User Choice/Control', 8),
                     ('Introductory/Generic', 9),
                     ('Practice not covered', 10),
                     ('Privacy contact information', 11)]
        )

    # Following lines ensure that the third column
    # from the CSV files are simply taken for the
    # labels actually present.
    files = glob.glob(os.path.join(path, '*.csv'))
    attributes = set()
    for f in files:
        for line in open(f, 'r'):
            line = line.strip()
            line = line.split('","')
            if is_parent:
                line = line[-1]
            else:
                line = line[2]
            line = line.replace('"','')
            attributes.add(line)

    labels = OrderedDict()
    for index, item in enumerate(attributes):
        labels[item] = index
        
    return labels
