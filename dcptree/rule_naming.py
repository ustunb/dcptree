# Author: Berk Ustun | www.berkustun.com
# License: BSD 3 clause

import re
from math import ceil, floor

RELATION_TYPES = {
    'eq',
    'neq',
    'lt',
    'gt',
    'geq',
    'leq',
    'is',
    'isnot',
    'in',
    'notin'
}

RELATION_COMPLEMENTS = {
    'eq': 'neq',
    'neq': 'eq',
    'lt': 'geq',
    'gt': 'leq',
    'geq': 'lt',
    'leq': 'gt',
    'is': 'isnot',
    'isnot': 'is',
    'in': 'notin',
    'notin': 'in'
}

SEPARATOR = '::'
RELATION_NAMES = {s:('%s%s%s' % (SEPARATOR, s, SEPARATOR)) for s in RELATION_TYPES}
COMPLEMENT_NAMES = {k:RELATION_NAMES[RELATION_COMPLEMENTS[k]] for k in RELATION_NAMES}

#rules from a boolean feature (e.g. Male)
RULE_PATTERN_BOOLEAN = '^(?!%s)\w+' % SEPARATOR
RULE_PARSER_BOOLEAN = re.compile(RULE_PATTERN_BOOLEAN)
BOOLEAN_COMPLEMENT_HEADER = "Not"

#generic rule pattern (applies to anything with a relation symbol)
NAME_PATTERN = '.+'
REL_PATTERN = "|".join(['%s' % s for s in RELATION_NAMES.keys()])
VALUE_PATTERN = '.+'
RULE_PATTERN = '(%s)%s(%s)%s(%s)' % (NAME_PATTERN, SEPARATOR, REL_PATTERN, SEPARATOR, VALUE_PATTERN)
RULE_PARSER = re.compile(RULE_PATTERN)

#values of categorical features
SET_SEPARATOR = ","
SET_START = "{"
SET_END = "}"
ITEM_PATTERN = "\w+"
SET_CONTENTS = "(%s)(%s%s)+" % (ITEM_PATTERN, SET_SEPARATOR, ITEM_PATTERN)
VALUE_PATTERN_CATEGORICAL = '%s%s%s' % (SET_START, SET_CONTENTS, SET_END)
VALUE_PARSER_CATEGORICAL = re.compile(VALUE_PATTERN_CATEGORICAL)

RULE_PATTERN_CATEGORICAL = '(%s)%s(%s)%s(%s)' % (NAME_PATTERN, SEPARATOR, REL_PATTERN, SEPARATOR, VALUE_PATTERN_CATEGORICAL)
RULE_PARSER_CATEGORICAL =re.compile(RULE_PATTERN_CATEGORICAL)

VALUE_PATTERN_ONE_CATEGORY = '(%s)|%s(%s)%s' % (ITEM_PATTERN, SET_START, ITEM_PATTERN, SET_END)
VALUE_PARSER_ONE_CATEGORY = re.compile(VALUE_PATTERN_ONE_CATEGORY)

#bin parsing
BIN_SEPARATOR = ','
BIN_START = '[\[|\(]'
BIN_END = '[\]|\)]'
BIN_INFINITY_PATTERN = '[-|+]?[iI][nN][fF]'
BIN_NUMBER_PATTERN = '[-|+]?[\d]+[\.[\d]*]?|[-]?\.[\d]+?'

BIN_PATTERN = '(%s)(%s|%s)%s(%s|%s)(%s)' % (BIN_START,
                                            BIN_NUMBER_PATTERN,
                                            BIN_INFINITY_PATTERN,
                                            BIN_SEPARATOR,
                                            BIN_NUMBER_PATTERN,
                                            BIN_INFINITY_PATTERN,
                                            BIN_END)

BIN_PARSER = re.compile(BIN_PATTERN)

#values of numeric features
VALID_INFINITY_VALUES = (float('-inf'), float('inf'))
VALID_BIN_TYPES = ['float','int']
VALID_DELIMITERS_START = ("[", "(")
VALID_DELIMITERS_END = (")", "]")

#ordinal values
ORDINAL_RANGE_SEPARATOR = ',' #could also be to


#Rule Naming

def validate_rule_name(rule_name):
    """
    
    :param rule_name: 
    :return: 
    """
    if type(rule_name) is not str:
        raise ValueError('rule_name must be string (found type =%r)', type(rule_name))

    rule_name = rule_name.strip()

    if len(rule_name) == 0:
        raise ValueError('rule_name cannot be empty')

    return rule_name


def parse_feature_name(rule_name):
    """
    extracts the name of a feature from a rule generated from that feature
    rule_name = 'Age::geq::20' -> Age
    rule_name = 'Age::is::1' -> Age
    rule_name = 'Male' -> Male
    :param rule_name:   non-empty string containing the name of a rule
                        must adhere to the format described in RULE_PATTERN
    :return: name of the feature used to generate the rule
    """

    rule_name = validate_rule_name(rule_name)
    parsed = RULE_PARSER.match(rule_name)
    if parsed is None: #occurs only if rule_name is Boolean or do
        parsed = RULE_PARSER_BOOLEAN.match(rule_name)
        if parsed is not None:
            parsed.group()
        else:
            raise ValueError('invalid rule_name')
    else:
        return parsed.groups()[0]


def validate_category_string(category_string):
    """
    :param category_string: non-empty string containing a subset of categories
    :return: 
    """

    if type(category_string) is not str:
        raise ValueError('category_name must be string (found type =%r)', type(category_string))

    category_string = "".join(category_string.split())

    if len(category_string) == 0:
        raise ValueError('rule_name cannot be empty')

    return category_string


def parse_categories(category_string):
    """
    """
    category_string = validate_category_string(category_string)
    parsed = VALUE_PARSER_CATEGORICAL.match(category_string)
    if parsed is None:
        parsed = VALUE_PARSER_ONE_CATEGORY.match(category_string)
        if parsed is None:
            raise ValueError("invalid category name")
        else:
            parsed_categories = [c for c in parsed.groups() if c is not None]
            return parsed_categories[0]
    else:
        parsed_categories = parsed.groups()
        parsed_categories = [c.strip(SET_SEPARATOR) for c in parsed_categories if c is not None]
        parsed_categories = set(parsed_categories)

    return parsed_categories


def parse_feature_value(rule_name):
    """
    returns the value of the feature from the name of a rule
    :param rule_name:   non-empty string containing the name of a rule
                        must adhere to the format described in RULE_PATTERN
    :return: value of the feature used to generate the rule
    """
    rule_name = validate_rule_name(rule_name)
    parsed = RULE_PARSER.match(rule_name)
    if parsed is None: #occurs only if rule_name is Boolean or do
        parsed = RULE_PARSER_BOOLEAN.match(rule_name)
        if parsed is not None:
            return True, 'boolean'
        else:
            raise ValueError('invalid rule_name')
    else:
        parsed_value_string = parsed.groups()[-1]

    #parsed value can not either be:
    #1
    #1.0
    #[-1.0,2]
    #[-inf, 10.10]
    #string

    #if parsed value is a single number then return the number
    try:
        parsed_value = float(parsed_value_string)
        parsed_type = 'number'
        if parsed_value.is_integer() and '.' not in parsed_value_string:
            return parsed_type, int(parsed_value)
        else:
            return parsed_type, parsed_value
    except ValueError:
        pass

    # if parsed value is a bin then return the bin
    try:
        parsed_value = parse_bin_elements(parsed_value_string)
        (start_delimiter, start_value, end_value, end_delimiter) = parsed_value
        if start_value == end_value and (start_delimiter, end_delimiter) == ('[',']'):
            parsed_value = start_value
            parsed_type = 'number'
        else:
            parsed_type = 'bin'
        return parsed_type, parsed_value
    except ValueError:
        pass

    #if parsed value is a string then return the string
    try:
        parsed_value = parse_categories(parsed_value_string)
        parsed_type = type(parsed_value)
        return parsed_type, parsed_value
    except ValueError:
        raise ValueError('could not parse value')


def convert_rule_name_to_feature_category(rule_name):
    parsed_type, parsed_value = parse_feature_value(rule_name)
    if parsed_type is 'bin':
        closure = parsed_value[0] + parsed_value[3]
        start = parsed_value[1]
        end = parsed_value[2]
        if float(start).is_integer() and float(end).is_integer():
            b = IntegerBin(start, end, closure)
        else:
            b = Bin(start, end, closure)
        category_label = str(b)
    else:
        category_label = '['+ str(parsed_value) + ']'
    return category_label


def parse_feature_relation(rule_name):
    """
    returns the relation between a feature and the rule used to generate that feature
    :param rule_name:   non-empty string containing the name of a rule
                        must adhere to the format described in RULE_PATTERN
    :return: 
    """
    parsed = RULE_PARSER.match(rule_name)
    if parsed is not None:
        return parsed.groups()[1]
    else:
        return rule_name.partition('::')[-1].partition('::')[0]


def numeric_name_to_ordinal_name(rule_name, vals_to_lvls):

    (name, symbol, _) = RULE_PARSER.match(rule_name).groups()
    value, value_type = parse_feature_value(rule_name)
    assert value_type in ('number', 'bin'), 'rule name is not numeric'

    #if value is one level (single_sided rule)
    if value_type is 'number':
        value = int(value)
        level = vals_to_lvls[int(value)]
        if symbol == 'eq':
            ordinal_name = '%s%s%s%s%s' % (name, SEPARATOR, 'is', SEPARATOR, level)
        elif symbol == 'neq':
            ordinal_name = '%s%s%s%s%s' % (name, SEPARATOR, 'isnot', SEPARATOR, level)
        else:
            ordinal_name = '%s%s%s%s%s' % (name, SEPARATOR, symbol, SEPARATOR, level)

    elif value_type is 'bin':

        (start_delimiter, start_value, end_value, end_delimiter) = value
        start_level = vals_to_lvls[int(start_value)]
        end_level = vals_to_lvls[int(end_value)]
        ordinal_level = start_delimiter + start_level + ORDINAL_RANGE_SEPARATOR + end_level + end_delimiter
        if symbol == 'eq':
            ordinal_name = '%s%s%s%s%s' % (name, SEPARATOR, 'is', SEPARATOR, ordinal_level)
        elif symbol == 'neq':
            ordinal_name = '%s%s%s%s%s' % (name, SEPARATOR, 'isnot', SEPARATOR, ordinal_level)
        else:
            ordinal_name = '%s%s%s%s%s' % (name, SEPARATOR, symbol, SEPARATOR, ordinal_level)

    return ordinal_name


def ordinal_bin_to_numeric_bin(ordinal_bin, ordering):
    (start_delimiter, start, end, end_delimiter) = parse_bin_elements_ordinal(bin_string=ordinal_bin, ordering=ordering)
    numeric_bin = start_delimiter + str(start) + ORDINAL_RANGE_SEPARATOR + str(end) + end_delimiter
    return numeric_bin


def get_complement_name(rule_name):
    #todo better handling of numeric, ordinal, categorical types
    #numeric: trim edges/ single_sided rules
    #ordinal: trim edges/ single_sided_rules
    #categorical use is if set is 1
    rule_name = validate_rule_name(rule_name)
    parsed = RULE_PARSER.match(rule_name)

    if parsed is not None:

        (feature_name, relation, feature_value) = parsed.groups()
        comp_name = '%s%s%s' % (feature_name, COMPLEMENT_NAMES[relation], feature_value)

    else:

        parsed = RULE_PARSER_BOOLEAN.match(rule_name)
        if parsed is None:
            raise ValueError('invalid rule_name')
        comp_name = BOOLEAN_COMPLEMENT_HEADER + parsed.group()

    return comp_name


# Bin Parsing

def validate_bin_string(bin_string):

    if type(bin_string) is not str:
        raise ValueError('invalid bin_string; bin_string must be a string')

    bin_string = "".join(bin_string.split())

    if len(bin_string) == 0:
        raise ValueError('invalid bin_string; bin_string must be non-empty')

    if BIN_SEPARATOR not in bin_string:
        raise ValueError('invalid bin_string; bin_string must contain two end points separated by "%s"'% BIN_SEPARATOR)

    return bin_string


def parse_bin(bin_string, bin_type = None):
    """
    Creates a Bin object from a valid Bin string.
    :param bin_string:          string with the form 
                                "S start, end E"  where:
                                - S is "[" or "(" 
                                - E is "]" or ")" 
                                - start, end are numbers such as 2, 2.0, .2, -2.
                                Whitespace in the bin_string is ignored
    :param bin_type (optional): Set as either 'float', 'int' 
                                By default, all bins have bin_type = 'float' if start or end contains a decimal point  
                                To override this, either include decimals in the bin string or specify bin_type
                                Note that it is possible to return a Bin over integers even when start, end are real 
                                numbers
    :return: Bin object initialized from the string if return_values = True
    """
    if bin_type not in (None, 'float', 'int'):
        raise ValueError('invalid bin type; bin_type must be unspecified or set as "float" or "int"')

    bin_string = validate_bin_string(bin_string)
    parsed_string = BIN_PARSER.match(bin_string)
    if parsed_string is None:
        raise ValueError('tried to parse invalid bin_string')

    (start_delimiter, start, end, end_delimiter) = parsed_string.groups()

    # detect bin type is unset
    if bin_type is None:
        bin_type = 'float' if ('.' in start or '.' in end) else 'int'

    #convert start and end values
    bin_start, bin_end = float(start), float(end)

    if bin_start == float('inf'):
        raise ValueError('invalid start point (cannot start bin at infinity')

    if bin_end == float('-inf'):
        raise ValueError('invalid end point (cannot end bin at -infinity')

    if bin_start > bin_end:
        raise ValueError('bin has invalid end points (start: %r > end: %r)' % parsed_string[1:3])

    #infer bin type from start/end String values
    if (bin_start, bin_end) == (float('-inf'), float('inf')):
        bin_type = 'float'

    return Bin(start = bin_start, end = bin_end, closure = start_delimiter + end_delimiter, type = bin_type)


def parse_bin_elements(bin_string, bin_type = None):
    """
    Creates a Bin object from a valid Bin string.
    :param bin_string:          string with the form 
                                "S start, end E"  where:
                                - S is "[" or "(" 
                                - E is "]" or ")" 
                                - start, end are numbers such as 2, 2.0, .2, -2.
                                Whitespace in the bin_string is ignored
    :param bin_type (optional): Set as either 'float', 'int' 
                                By default, all bins have bin_type = 'float' if start or end contains a decimal point  
                                To override this, either include decimals in the bin string or specify bin_type
                                Note that it is possible to return a Bin over integers even when start, end are real 
                                numbers
    :return: Bin object initialized from the string if return_values = True
    """
    if bin_type not in (None, 'float', 'int'):
        raise ValueError('invalid bin type; bin_type must be unspecified or set as "float" or "int"')

    bin_string = validate_bin_string(bin_string)
    parsed_string = BIN_PARSER.match(bin_string)
    if parsed_string is None:
        raise ValueError('tried to parse invalid bin_string')

    (start_delimiter, start, end, end_delimiter) = parsed_string.groups()

    # detect bin type is unset
    if bin_type is None:
        bin_type = 'float' if ('.' in start or '.' in end) else 'int'

    # convert start and end values
    bin_start,  bin_end = float(start), float(end)

    # return integers is bin type is over integers
    if bin_type == 'int':
        if bin_start.is_integer():
            bin_start = int(bin_start)
        if bin_end.is_integer():
            bin_end = int(bin_end)

    return start_delimiter, bin_start, bin_end, end_delimiter


def parse_bin_elements_ordinal(bin_string, ordering):

    level_pattern = "|".join(['%s' % l for l in ordering])
    value_pattern = "|".join(['%s' % s for s in range(len(ordering))])
    ordinal_bin_pattern = '(%s)(%s|%s|%s)%s(%s|%s|%s)(%s)' % (BIN_START,
                                                              level_pattern,
                                                              value_pattern,
                                                              BIN_INFINITY_PATTERN,
                                                              BIN_SEPARATOR,
                                                              level_pattern,
                                                              value_pattern,
                                                              BIN_INFINITY_PATTERN,
                                                              BIN_END)

    bin_string = validate_bin_string(bin_string)
    parsed_string = re.match(pattern = ordinal_bin_pattern, string = bin_string)
    if parsed_string is None:
        raise ValueError('tried to parse invalid bin_string')

    (start_delimiter, start_level, end_level, end_delimiter) = parsed_string.groups()

    # convert start and end values
    if start_level in ordering:
        bin_start = ordering.index(start_level)
    else:
        bin_start = float(start_level)
        bin_start = int(bin_start) if bin_start.is_integer() else bin_start

    if end_level in ordering:
        bin_end = ordering.index(end_level)
    else:
        bin_end = float(end_level)
        bin_end = int(bin_end) if bin_end.is_integer() else bin_end

    if bin_start == float('inf'):
        raise ValueError('invalid start point (cannot start bin at infinity')

    if bin_end == float('-inf'):
        raise ValueError('invalid end point (cannot end bin at -infinity')

    if bin_start > bin_end:
        raise ValueError('bin has invalid end points (start: %r > end: %r)' % parsed_string[1:3])

    return start_delimiter, bin_start, bin_end, end_delimiter


# Bins
class Bin(object):
    """
    Wrapper class for Bin objects over real numbers and integers 
    By default, all bins are over real numbers
    """
    def __init__(self, start, end, closure='[)', type='float'):

        # remove whitespaces
        closure = "".join(closure.split()).lower()

        # wrap the object
        if start > end:
            raise ValueError('invalid end points; received end points (start, end) where start > end')

        #basic error checking
        if len(closure) != 2:
            raise ValueError('invalid closure for bin; closure must be one of "[]","(]", "[)", "()"')

        if closure[0] not in VALID_DELIMITERS_START:
            raise ValueError('invalid start delimiter in closure')

        if closure[1] not in VALID_DELIMITERS_END:
            raise ValueError('invalid end delimiter in closure')

        if type not in ('float', 'int'):
            raise ValueError('invalid bin type (%r); bin type must either be "int" or "float"' % type)

        self._wrapped_obj = FloatBin(start, end, closure) if type is 'float' else IntegerBin(start, end, closure)
        self.type = type

    def __repr__(self):
        return self._wrapped_obj.__repr__()

    def __len__(self):
        return self._wrapped_obj.__len__()

    def __eq__(self, other):
        return self._wrapped_obj.__eq__(other)

    def __contains__(self, other):
        return self._wrapped_obj.__contains__(other)

    def __getattr__(self, attr):
        # see if this object has attr
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self._wrapped_obj, attr)

    def contains(self, arr):
        """
        :param arr: 
        :return: boolean vector
        """
        return [self._wrapped_obj.__contains__(v) for v in arr]

    def rule_string(self, for_complement = False):
        """
        :param symbol_set: set of symbols containing at a minimum 'eq', 'gt', 'lt' 'leq', 'geq', 'gt'
        :return: informative string that can be used to name a rule when this bin is used to 
        produce a binary rule from a feature   
        """
        sym_string = RELATION_NAMES[self.comp_symbol] if for_complement else RELATION_NAMES[self.symbol]
        val_string = self.threshold_string if self.one_sided else self._wrapped_obj.__repr__()
        return sym_string + val_string


class FloatBin(object):
    """
    represents a non-empty interval over the reals
    closure can be one of: 
    '()' open
    '[]' closed
    '(]' left half-open 
    '[)' right half-open
    """

    def __init__(self, start, end, closure='[)'):

        if start > end:
            raise ValueError('invalid end points; received end points (start, end) where start > end')

        if closure not in ('(]', '()', '[]', '[)'):
            raise ValueError('invalid closure for bin; closure must be one of "[]","(]", "[)", "()"')

        if start == end and closure is not '[]':
            raise ValueError('if start == end, then closure must be set as "[]" for non-empty bin')

        self.start = float(start)
        self.end = float(end)
        self.closure = closure

        self.open_at_start = self.closure[0] == '('
        self.open_at_end = self.closure[1] == ')'
        self.closed_at_start = not self.open_at_start
        self.closed_at_end = not self.open_at_end

        if self.start == float('-inf') and self.end < float('inf'):
            self.one_sided = True
            self.threshold_value = self.end
            if closure[1] == ']':
                self.symbol, self.comp_symbol = 'leq', 'gt'
            else:
                self.symbol, self.comp_symbol = 'lt', 'geq'
        elif self.start > float('-inf') and self.end == float('inf'):
            self.one_sided = True
            self.threshold_value = self.start
            if closure[0] == '[':
                self.symbol, self.comp_symbol = 'geq', 'lt'
            else:
                self.symbol, self.comp_symbol = 'gt', 'leq'
        else:
            self.one_sided = False
            self.threshold_value = float('nan')
            self.symbol, self.comp_symbol = 'in','notin'

        self.threshold_string = '%f' % self.threshold_value
        self.__check_rep__()

    def __len__(self):
        """@return interval length from start to end"""
        return (self.end - self.start)

    def __repr__(self):
        """@return String rep"""
        [l_str, r_str] = list(self.closure)
        return '%s%r,%r%s' % (l_str, self.start, self.end, r_str)

    def __eq__(self, other):
        """@return: True iff bins are subsets of one another"""
        return self.start == other.start and self.end == other.end and self.closure == other.closure

    def __contains__(self, value):
        """@return: True iff interval contains float(value)"""
        value = float(value)
        if self.start < value < self.end:
            return True
        elif self.start == value:
            return bool(self.closed_at_start)
        elif self.end == value:
            return bool(self.closed_at_end)
        else:
            return False

    def __check_rep__(self):
        assert self.start <= self.end
        if self.start < self.end:
            assert self.__len__() > 0
        else:
            assert len(self) == 0
        return True


class IntegerBin(object):
    """
    returns the largest closed interval of integers contained within a non-empty interval of real numbers
    valid inputs consist of:
    - start (int or float), will be rounded up if float
    - end (int or float), will be rounded down if float
    - closure, which must either be: 
        '()' open
        '[]' closed
        '(]' left half-open 
        '[)' right half-open 
    """

    def __init__(self, start, end, closure='[)'):

        if start > end:
            raise ValueError('invalid end points; received end points (start, end) where start > end')

        if closure not in ('(]', '()', '[]', '[)'):
            raise ValueError('invalid closure for bin; closure must be one of "[]","(]", "[)", "()"')

        if float(start) in VALID_INFINITY_VALUES:
            closure = '(' + closure[1]
        elif not float(start).is_integer():
            start = ceil(start)
            closure = '[' + closure[1]
        else:
            start = int(start)
            if closure[0] in '(':
                start += 1
                closure = '[' + closure[1]

        if float(end) in VALID_INFINITY_VALUES:
            closure = closure[0] + ')'
        elif not float(end).is_integer():
            end = floor(end)
            closure = closure[0] + ']'
        else:
            end = int(end)
            if closure[1] in ')':
                end -= 1
                closure = closure[0] + ']'

        self.start = start
        self.end = end
        self.closure = closure
        self.open_at_start = self.closure[0] == '('
        self.open_at_end = self.closure[1] == ')'
        self.closed_at_start = not self.open_at_start
        self.closed_at_end = not self.open_at_end

        if self.start == float('-inf') and self.end < float('inf'):
            self.one_sided = True
            if closure[1] == ']':
                self.symbol, self.threshold_value, self.comp_symbol = 'leq', self.end, 'gt'
            else:
                self.symbol, self.threshold_value, self.comp_symbol = 'lt', self.end+1,  'geq'
        elif self.start > float('-inf') and self.end == float('inf'):
            self.one_sided = True
            if closure[0] == '[':
                self.symbol, self.threshold_value, self.comp_symbol = 'geq', self.start, 'lt'
            else:
                self.symbol, self.threshold_value, self.comp_symbol = 'gt', self.start+1, 'leq'
        else:
            self.one_sided = False
            if self.start < self.end:
                self.symbol, self.comp_symbol = 'in', 'notin'
            else:
                self.symbol, self.comp_symbol = 'eq', 'noteq'

        self.threshold_string = '%d' % self.threshold_value if self.one_sided else ''
        self.__check_rep__()

    def __len__(self):
        return self.end - self.start + 1

    def __repr__(self):
        if self.start == self.end:
            return '%r' % self.start
        else:
            return '[%r,%r]' % (self.start, self.end)

    def __eq__(self, other):
        """@return: True iff bins are subsets of one another"""
        return self.start == other.start and self.end == other.end and self.closure == other.closure

    def __contains__(self, value):
        """@return: True iff interval contains float(value)"""
        value = float(value)
        if self.start < value < self.end:
            return True
        elif self.start == value:
            return bool(self.closed_at_start)
        elif self.end == value:
            return bool(self.closed_at_end)
        else:
            return False

    def __check_rep__(self):
        assert self.start <= self.end
        assert self.__len__() >= 1
        assert float(self.start).is_integer() or self.start in (float('-inf'), float('inf'))
        assert float(self.end).is_integer() or self.end in (float('-inf'), float('inf'))
        return True