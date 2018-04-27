class Helper:
    def handle_forward_hook(module, input, output):
        print('***********forward_hook***************')
        print(module)
        print('Forward Input', input)
        print('Output Output', output)
        print('**************************')

    def handle_backward_hook(module, input, output):
        print('***********backward_hook***************')
        print(module)
        print('Grad Input', input)
        print('Grad Output', output)
        print('**************************')

    def handle_variable_hidden_hook(grade):
        print('***********hidden_hook***************')
        # grade.data[0][0] = 0.0
        # grade.data[0][1] = 0.0
        print('grade: ', grade)
        # grade.data[0] = 0
        print('**************************')

    def handle_variable_predict_hook(grade):
        print('***********predict_hook***************')
        print('grade: ', grade)
        # modify
        # grade.data[0] = 0
        print('**************************')


