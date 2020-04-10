"""
The functions in this file are originated from the code for
Compound Probabilistic Context-Free Grammars for Grammar Induction,
Y. Kim et al., ACL 2019.
For more details, visit https://github.com/harvardnlp/compound-pcfg.
"""

import re


def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w


def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn


def get_nonbinary_spans(actions, SHIFT=0, REDUCE=1):
    spans = []
    tags = []
    stack = []
    pointer = 0
    binary_actions = []
    nonbinary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            nonbinary_actions.append(SHIFT)
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == 'NT(':
            # stack.append('(')
            stack.append(action[3:-1].split('-')[0])
        elif action == "REDUCE":
            nonbinary_actions.append(REDUCE)
            right = stack.pop()
            left = right
            n = 1
            # while stack[-1] is not '(':
            while type(stack[-1]) is tuple:
                left = stack.pop()
                n += 1
            span = (left[0], right[1])
            tag = stack.pop()
            if left[0] != right[1]:
                spans.append(span)
                tags.append(tag)
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)
                num_reduce += 1
        else:
            assert False
    assert (len(stack) == 1)
    assert (num_shift == num_reduce + 1)
    return spans, tags, binary_actions, nonbinary_actions


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i):  # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1
                # get the next open bracket,
                # which may be a terminal or another non-terminal
                while line_strip[i] != '(':
                    i += 1
            else:  # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
            output_actions.append('REDUCE')
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != ')' and line_strip[i] != '(':
                i += 1
    assert i == max_idx
    return output_actions


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, '
                     'open bracket not followed by closed bracket')


def get_nonterminal(line, start_idx):
    assert line[start_idx] == '('  # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not (char == '(') and not (char == ')')
        output.append(char)
    return ''.join(output)


def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        # fulfilling this condition means this is a terminal symbol
        if line_strip[i] == '(' and not (is_next_open_bracket(line_strip, i)):
            output.append(get_between_brackets(line_strip, i))
    # print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        # print(terminal, terminal_split)
        assert len(
            terminal_split) == 2  # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not (char == '(')
        output.append(char)
    return ''.join(output)
