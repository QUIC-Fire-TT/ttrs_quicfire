# Contributing to ttrs_quicfire

The following is a set of guidelines for contributing to the ttrs_quicfire
library. These are guidelines, not rules. However, sticking to them will
be beneficial to keeping the repo clean and stable.

## I Just Have a Question!
The best way to get your question answered is to email Zach (zcope@talltimbers.org)
or Cameron (cambones12@gmail.com). <You should also check the FAQs>
<before emailing.> Please only open an issue for 
[bugs](https://github.com/QUIC-Fire-TT/ttrs_quicfire/blob/main/CONTRIBUTING.md#Reporting-Bugs)
or [new feature requests](https://github.com/QUIC-Fire-TT/ttrs_quicfire/blob/main/CONTRIBUTING.md#Feature-Requests).

## How Can I Contribute?

### Reporting Bugs
This section will give you guidelines on how to report a bug. Following 
these guidelines will make it easier for us to understand and reproduce 
your issue.

Before submitting a new bug report, check to see if one has already been 
created. You can do this by searching 
[here](https://github.com/QUIC-Fire-TT/ttrs_quicfire/issues).

> **NOTE:** If a closed issue is close to what you are reporting, create
a new report and link the closed report.

#### How do I Submit a Good Bug Report?
When writing your bug report use this 
[template](https://github.com/QUIC-Fire-TT/ttrs_quicfire/issues/new?assignees=&labels=&template=bug_report.md&title=)
and try and include the following things:

* **Use a clear and descriptive title.**
* **Describe the exact steps to reproduce the probelm** in as much detail 
as possible. For example, included which shapefiles you are using, your
specific function calls, parameters, etc.
* **Provide code snippets for the problem** so we can easily reproduce your code.
* **Provide details of the behavior you are seeing.**
* **Provide what you expected to happen and why.**
* **Include any screenshots that may help.**

Include this information about your configuration:

* **ttrs_quicfire version.** This can be gathered by using 
```ttrs_quicfire.__VERSION__``` in your python interpreter. 
* **Operating System** ex. macOS Mojave
* **Operating System Version** ex. 10.14.6

### Feature Requests
This section will give you guidelines on how to make a feature request.
Following these guidelines will make it easier for us to under your request
and implement it.

Before submitting a new request, check to make sure this feature has not been
requested or has been implemented. Current requests can be searched 
[here](https://github.com/QUIC-Fire-TT/ttrs_quicfire/issues). Current implementation
can be found in the API reference 
[here](https://github.com/QUIC-Fire-TT/ttrs_quicfire/wiki/API-Reference).
> **NOTE:** If a closed issue is similar to request, please link it when creating the
new one.

#### How do I Submit a Good Feature Request?
When writing your feature request use this 
[template](https://github.com/QUIC-Fire-TT/ttrs_quicfire/issues/new?assignees=&labels=&template=feature_request.md&title=)
and try and include the following things:

* **Use a clear and descriptive title.**
* **Provide a step-by-step description of the feature** in as much detail as possible.
* **Provide code snippets for the feature** that you already have or that may be useful
in the code that already exists.
* **Provide details on your intended output for the feature.**
* **Include any screenshots that may help.**


### Code Contributions
To get started, follow the 
[installation instructions](https://github.com/QUIC-Fire-TT/ttrs_quicfire/wiki/Installation).
Once you have the library up and running, you can copy the code to your local
machine in your normal development workspace. Once you have the code copied you
can start writing code and test it. In order to test your code, you need to setup a project
You can find instructions for setting up a project
[here](https://github.com/QUIC-Fire-TT/ttrs_quicfire/wiki/ttrs_quicfire-Quick-Start).

When you make changes you will need to use ```pip install``` for them to work locally. 
From your local repository, run the following command each time a change is made:
```console
(fastfuels) system:ttrs_quicfire user$ pip install .
```

<!-- The second way is to use ```setup tools``` development mode. To set this up, run the
following command in your local repository:
```console
(fastfuels) system:ttrs_quicfire user$ setup.py develop
```

When you are done, you can use this command to switch out of develop mode and use the
installed version of ttrs_quicfire:
```console
(fastfuels) system:ttrs_quicfire user$ setup.py develop --uninstall
``` -->

Be sure to utilize git's branch system as you won't be able to push to the ```main```
branch without a [pull request](https://github.com/QUIC-Fire-TT/ttrs_quicfire/blob/main/CONTRIBUTING.md#Pull-Requests). 

### Pull Requests
This section will outline how to create a pull request for this repository.
Following these guidelines will make it easier for us to under your changes
and will increase the likelihood of an accepted request.

#### How do I Submit a Good Pull Request?
You can open a pull request [here](https://github.com/QUIC-Fire-TT/ttrs_quicfire/pulls).
When writing your pull request use this 
[template](https://github.com/QUIC-Fire-TT/ttrs_quicfire/tree/main/.github/pull_requset_template.md)
and try and include the following things:

* **Use a clear and descriptive title.**
* **Provide a step-by-step description of the feature** in as much detail as possible.
* **Provide the context and motivation of the feature.** Why it was made and why it
is useful to the user.
* **Provide details on the intended output for the feature** along with the test cases.
* **Include any screenshots that may help.**

Finally, ensure the checklist is complete and you are ready to make the pull request!
It will be reviewed shortly after you submitted and will either be accepted or
we will ask for clarification or fixes if anything is confusing or broken.